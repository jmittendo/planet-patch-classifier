# This file is part of planet-patch-classifier, a Python tool for generating and
# classifying planet patches from satellite imagery via unsupervised machine learning
# Copyright (C) 2024  Jan Mittendorf (jmittendo)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import typing
from datetime import datetime
from pathlib import Path

import einops
import numpy as np
import spiceypy as spice
from pandas import DataFrame
from PIL import Image
from planetaryimage import PDS3Image
from pvl import PVLModule, Quantity
from tqdm import tqdm

import source.satellite_dataset.patches.utility as sd_patch_util
import source.satellite_dataset.utility as sd_util
import user.config as user_config

if typing.TYPE_CHECKING:
    from source.satellite_dataset import SatelliteDataset


# see SPICE ik kernel "juno_junocam_v03.ti"
FILTER_NAME_IDS = {"BLUE": 1, "GREEN": 2, "RED": 3, "METHANE": 4}
PHOTOACTIVE_RANGE = (23, 1631)
FRAMELET_HEIGHT = 128

PLANET = "JUPITER"
SPACECRAFT = "JUNO"
PLANET_FRAME = "IAU_JUPITER"
CAMERA_FRAME = "JUNO_JUNOCAM"
SURFACE_NORMAL_METHOD = "ELLIPSOID"


class JnoJncPatchGenerator:
    def __init__(self, patch_scale_km: float, patch_resolution: int) -> None:
        self._patch_scale_km = patch_scale_km
        self._patch_resolution = patch_resolution

    def generate(self, dataset: "SatelliteDataset", output_dir_path: Path) -> None:
        spice_path = dataset.archive.spice_path

        if spice_path is None:
            raise ValueError("Spice path must not be 'None' for 'jno-jnc' archive")

        sd_util.load_spice_kernels(spice_path)

        version_dir_path = output_dir_path / "versions" / "norm-local"
        version_dir_path.mkdir(parents=True, exist_ok=True)

        ellipsoid_radii = get_body_radii(PLANET)

        # Step 1
        patch_indices = (
            np.mgrid[self._patch_resolution - 1 : -1 : -1, : self._patch_resolution]
            .transpose(1, 2, 0)
            .reshape(-1, 2)
        )
        patch_ray_origins = get_patch_ray_origins(
            self._patch_scale_km, self._patch_resolution, ellipsoid_radii[0]
        )
        patch_ray_directions = np.full_like(patch_ray_origins, [-1, 0, 0])

        # Step 2/3
        patch_theta_centers, patch_phi_centers = get_patch_theta_and_phi_centers(
            self._patch_scale_km, ellipsoid_radii.max()
        )

        # Precompute lists of intersection points and corresponding patch array indices
        # for every patch phi- and theta-center (patch indices are independent of phi)
        (
            theta_phi_intersection_points,
            theta_patch_indices,
        ) = get_theta_phi_intersection_points_and_patch_indices(
            patch_theta_centers,
            patch_phi_centers,
            patch_ray_origins,
            patch_ray_directions,
            patch_indices,
            ellipsoid_radii,
        )

        junocam_to_framelet_filter_params = get_junocam_to_framelet_filter_params(
            list(FILTER_NAME_IDS.values())
        )

        # Spatial resolution of a patch in m/px
        # (ignoring projection effects / distortions)
        patch_resolution_mpx = self._patch_scale_km * 1000 / self._patch_resolution

        patch_file_names: list[str] = []
        patch_longitudes: list[float] = []
        patch_latitudes: list[float] = []
        patch_local_times: list[float] = []

        for data in tqdm(dataset, desc="Generating patches"):
            max_resolution_mpx = data["max_resolution_mpx"]

            if not sd_patch_util.passes_resolution_threshold(
                max_resolution_mpx, patch_resolution_mpx
            ):
                continue

            pds3_lbl_path = data["pds3_lbl_path"]
            pds3_image = PDS3Image.open(pds3_lbl_path.resolve().as_posix())

            filter_names = pds3_image.label["FILTER_NAME"]  # type: ignore
            filter_ids = [FILTER_NAME_IDS[fn] for fn in filter_names]
            num_filters = len(filter_ids)
            frame_height = FRAMELET_HEIGHT * num_filters

            image_time_utc: datetime = pds3_image.label["IMAGE_TIME"]  # type: ignore
            image_time_et: float = spice.str2et(  # type: ignore
                image_time_utc.strftime(r"%Y-%m-%d %H:%M:%S.%f")
            )

            # Get all frames and corresponding frame times (in ephemeris time) in the
            # PDS3 image (one image consists of multiple frames taken at different times
            # while the spacecraft is spinning and one frame consists of num_filters
            # framelets)
            frames, frame_times_et = get_frames(pds3_image, frame_height)

            # Precompute framelets and ellipsoid positions/orientations for every frame
            frame_framelets = get_frame_framelets(frames, FRAMELET_HEIGHT)
            frame_ellipsoid_positions_and_orientations = (
                get_frame_ellipsoid_positions_and_orientations(
                    frame_times_et,
                    PLANET,
                    SPACECRAFT,
                    PLANET_FRAME,
                    CAMERA_FRAME,
                    abberation_correction=user_config.SPICE_ABBERATION_CORRECTION,
                )
            )
            frame_spacecraft_positions = get_frame_spacecraft_positions(
                SPACECRAFT,
                frame_times_et,
                PLANET,
                PLANET_FRAME,
                abberation_correction=user_config.SPICE_ABBERATION_CORRECTION,
            )

            file_index = 0

            for phi_intersection_points, patch_indices, theta_center in zip(
                theta_phi_intersection_points, theta_patch_indices, patch_theta_centers
            ):
                # Calculate geocentric latitude corresponding to theta center
                tan_theta_center = np.tan(theta_center)

                if tan_theta_center == 0:
                    latitude_center = 0.5 * np.pi
                else:
                    latitude_center = np.arctan(
                        ellipsoid_radii[2] / ellipsoid_radii[0] / tan_theta_center
                    )

                latitude_center = np.rad2deg(latitude_center)

                for intersection_points, phi_center in zip(
                    phi_intersection_points, patch_phi_centers
                ):
                    # Intialize the full patch image and overlap counter array
                    patch_image = np.zeros(
                        (self._patch_resolution, self._patch_resolution, num_filters)
                    )
                    patch_image_overlaps = patch_image.copy()

                    for (
                        framelets,
                        ellipsoid_position_and_orientation,
                        spacecraft_position,
                        frame_time_et,
                    ) in zip(
                        frame_framelets,
                        frame_ellipsoid_positions_and_orientations,
                        frame_spacecraft_positions,
                        frame_times_et,
                    ):
                        (
                            ellipsoid_position,
                            ellipsoid_orientation,
                        ) = ellipsoid_position_and_orientation

                        # Get the close and far intersection points for rays from the
                        # spacecraft to all points to perform the visibility check
                        visibility_rays = intersection_points - spacecraft_position

                        (
                            visibility_intersections_close,
                            visibility_intersections_far,
                        ) = get_ray_ellipsoid_intersection_points(
                            visibility_rays,
                            ellipsoid_radii,
                            ray_origins=spacecraft_position,
                            far_points=True,
                        )

                        # If the original points "correspond" (i.e. are closer) to the
                        # close intersection points, they are on the visible side of the
                        # ellipsoid, else they are not
                        distances_close = (
                            (visibility_intersections_close - intersection_points) ** 2
                        ).sum(axis=1)
                        distances_far = (
                            (visibility_intersections_far - intersection_points) ** 2
                        ).sum(axis=1)
                        visibility = np.where(
                            distances_close < distances_far, True, False
                        )

                        # Remove the invisible points and the corresponding patch array
                        # indices
                        points = intersection_points[visibility]
                        frame_patch_indices = patch_indices[visibility]

                        if points.size == 0:
                            continue

                        # Skip this frame if any corner pixel point does not pass the
                        # incidence angle threshold
                        corner_pixel_indices = find_corner_pixel_indices(
                            frame_patch_indices, self._patch_resolution
                        )

                        if corner_pixel_indices.size == 0:
                            continue

                        corner_pixel_points = points[corner_pixel_indices]

                        corner_point_incidence_angles = get_surface_point_solar_incidence_angles_deg(
                            corner_pixel_points,
                            PLANET,
                            PLANET_FRAME,
                            frame_time_et,
                            SURFACE_NORMAL_METHOD,
                            abberation_correction=user_config.SPICE_ABBERATION_CORRECTION,
                        )

                        if np.any(
                            corner_point_incidence_angles
                            > user_config.PATCH_ANGLE_THRESHOLD
                        ):
                            continue

                        # Transform points into the spacecraft camera frame
                        points = (
                            ellipsoid_orientation @ points.T
                        ).T + ellipsoid_position

                        for framelet, framelet_filter_id in zip(framelets, filter_ids):
                            # Transform point coordinates from 3d JunoCam frame into 2d
                            # framelet coordinates
                            framelet_coordinates = junocam_to_framelet_coordinates(
                                points,
                                junocam_to_framelet_filter_params[framelet_filter_id],
                            )

                            # Check whether points lie inside the photoactive
                            # framelet region
                            lower_bound = [PHOTOACTIVE_RANGE[0], 0]
                            upper_bound = [PHOTOACTIVE_RANGE[1], FRAMELET_HEIGHT]
                            inside_photoactive_region_mask = np.all(
                                (framelet_coordinates > lower_bound)
                                & (framelet_coordinates < upper_bound),
                                axis=1,
                            )

                            # Remove points that are ouside the photoactive region and
                            # the corresponding patch array indices
                            framelet_coordinates = framelet_coordinates[
                                inside_photoactive_region_mask
                            ]
                            framelet_patch_indices = frame_patch_indices[
                                inside_photoactive_region_mask
                            ]

                            # Convert framelet coordinates to integer pixel coordinates
                            # on the framelet
                            pixel_coordinates = framelet_coordinates.astype(int)

                            # Get pixel values from the framelet for every pixel
                            # coordinate
                            pixel_values = framelet[
                                pixel_coordinates[:, 1], pixel_coordinates[:, 0]
                            ]

                            # Create the patch image for this framelet
                            framelet_patch_image = np.zeros(
                                (self._patch_resolution, self._patch_resolution)
                            )
                            framelet_patch_image[*framelet_patch_indices.T] = (
                                pixel_values
                            )

                            # Add the framelet patch image to the total patch image in
                            # the corresponding filter and count possible overlaps
                            filter_index = filter_ids.index(framelet_filter_id)
                            patch_image[..., filter_index] += framelet_patch_image
                            patch_image_overlaps[..., filter_index] += (
                                framelet_patch_image > 0
                            )

                    # Don't save the patch image if any pixel was never assigned a value
                    if np.any(patch_image_overlaps == 0):
                        continue

                    min_image_value = pds3_image.image.min()  # type: ignore
                    max_image_value = pds3_image.image.max()  # type: ignore

                    patch_image = process_patch_image(
                        patch_image,
                        patch_image_overlaps,
                        max_image_value,
                        min_image_value,
                    )

                    pds3_file_name: str = pds3_image.label["FILE_NAME"]  # type: ignore

                    output_file_name = (
                        f"{pds3_file_name.split('.')[0].replace('_', '-')}"
                        f"_patch-{file_index}.png"
                    )

                    longitude_center = sd_util.fix_360_longitude(np.rad2deg(phi_center))

                    local_time_h, local_time_m, local_time_s, _, _ = spice.et2lst(
                        image_time_et,
                        spice.bodn2c(PLANET),
                        phi_center,
                        "PLANETOCENTRIC",
                    )
                    local_time_center = (
                        local_time_h + local_time_m / 60 + local_time_s / 3600
                    )

                    save_patch_image(patch_image, version_dir_path / output_file_name)
                    patch_file_names.append(output_file_name)
                    patch_longitudes.append(longitude_center)
                    patch_latitudes.append(latitude_center)
                    patch_local_times.append(local_time_center)

                    file_index += 1

        patch_info_table_dict = {
            "file_name": patch_file_names,
            "longitude": patch_longitudes,
            "latitude": patch_latitudes,
            "local_time": patch_local_times,
        }

        patch_info_table = DataFrame(data=patch_info_table_dict)
        patch_info_table.to_pickle(output_dir_path / "table.pkl")


def load_spice_kernels(kernels_dir: str) -> None:
    for file_path in Path(kernels_dir).rglob("*"):
        if file_path.is_file():
            if file_path.parent == "ck" and "sc_rec" not in file_path.name:
                continue

            if file_path.suffix == ".lbl":
                continue

            spice.furnsh(file_path.resolve().as_posix())


def get_junocam_to_framelet_filter_params(
    filter_ids: list[int],
) -> dict[int, dict[str, float]]:
    # See SPICE ik kernel "juno_junocam_v03.ti"

    filter_params: dict[int, dict[str, float]] = {}

    for filter_id in filter_ids:
        params: dict[str, float] = {}

        params["cx"] = spice.gdpool(f"INS-6150{filter_id}_DISTORTION_X", 0, 1)[0]
        params["cy"] = spice.gdpool(f"INS-6150{filter_id}_DISTORTION_Y", 0, 1)[0]
        params["k1"] = spice.gdpool(f"INS-6150{filter_id}_DISTORTION_K1", 0, 1)[0]
        params["k2"] = spice.gdpool(f"INS-6150{filter_id}_DISTORTION_K2", 0, 1)[0]
        params["fl"] = (
            spice.gdpool(f"INS-6150{filter_id}_FOCAL_LENGTH", 0, 1)[0]
            / spice.gdpool(f"INS-6150{filter_id}_PIXEL_SIZE", 0, 1)[0]
        )

        filter_params[filter_id] = params

    return filter_params


def get_body_radii(body_name: str) -> np.ndarray:
    return spice.bodvrd(body_name, "RADII", 3)[1]


def get_patch_ray_origins(
    patch_size: float, patch_resolution: int, equatorial_radius: float
) -> np.ndarray:
    # TODO: Speed improvedment:
    # Technically the first part of the algorithm is entirely symmetric with respect to
    # the x- and z- axis. Therefore only a quarter of the rays (maybe even an eighth) is
    # actually required.

    half_patch_size = 0.5 * patch_size
    slice_iterator = complex(imag=patch_resolution)

    origins_z, origins_y = np.mgrid[
        -half_patch_size:half_patch_size:slice_iterator,
        -half_patch_size:half_patch_size:slice_iterator,
    ]

    origins_x = np.full_like(origins_y, equatorial_radius)

    origins = np.stack([origins_x, origins_y, origins_z], axis=-1).reshape(-1, 3)

    return origins


def get_patch_theta_and_phi_centers(
    patch_size: float, max_ellipsoid_radius: float
) -> tuple[np.ndarray, np.ndarray]:
    if patch_size > max_ellipsoid_radius * 2:
        raise ValueError("Patch size cannot be larger than max ellipsoid radius.")

    num_phi_centers = int(
        np.ceil(np.pi / np.arcsin(patch_size / max_ellipsoid_radius / 2))
    )
    num_theta_centers = int(np.ceil(num_phi_centers / 2))

    theta_centers = np.linspace(0, np.pi, num_theta_centers)
    phi_centers = np.linspace(0, 2 * np.pi, num_phi_centers)

    return theta_centers, phi_centers


def get_theta_phi_intersection_points_and_patch_indices(
    patch_theta_centers: np.ndarray,
    patch_phi_centers: np.ndarray,
    patch_ray_origins: np.ndarray,
    patch_ray_directions: np.ndarray,
    patch_indices: np.ndarray,
    ellipsoid_radii: np.ndarray,
) -> tuple[list[list[np.ndarray]], list[np.ndarray]]:
    theta_phi_intersection_points = []
    theta_patch_indices = []

    for patch_theta in patch_theta_centers:
        # TODO: Because theta values are most likely symmetrical around 90°, half of
        # this loop could potentially be skipped

        geodetic_offset, geodetic_latitude = get_geodetic_origin_offset_and_latitude(
            patch_theta, ellipsoid_radii[0], ellipsoid_radii[2]
        )

        rotated_ray_origins = get_y_rotated_points(
            patch_ray_origins, -geodetic_latitude
        )
        rotated_ray_directions = get_y_rotated_points(
            patch_ray_directions, -geodetic_latitude
        )

        rotated_ray_origins[:, 0] += geodetic_offset

        intersection_points = get_ray_ellipsoid_intersection_points(
            rotated_ray_directions,
            ellipsoid_radii,
            ray_origins=rotated_ray_origins,
            far_points=False,
        )

        invalid_mask = np.any(np.isnan(intersection_points), axis=1)
        valid_intersection_points = intersection_points[~invalid_mask]
        valid_patch_indices = patch_indices[~invalid_mask]

        phi_intersection_points = [
            get_z_rotated_points(valid_intersection_points, patch_phi)
            for patch_phi in patch_phi_centers
        ]

        theta_phi_intersection_points.append(phi_intersection_points)
        theta_patch_indices.append(valid_patch_indices)

    return theta_phi_intersection_points, theta_patch_indices


def get_geodetic_origin_offset_and_latitude(
    parametric_theta: float, equatorial_radius: float, polar_radius: float
) -> tuple[float, float]:
    origin_offset = np.sin(parametric_theta) * (
        equatorial_radius - polar_radius**2 / equatorial_radius
    )

    tan_theta = np.tan(parametric_theta)

    if tan_theta == 0:
        latitude = 0.5 * np.pi
    else:
        latitude = np.arctan(equatorial_radius / polar_radius / tan_theta)

    return origin_offset, latitude


def get_z_rotated_points(point_coordinates: np.ndarray, angle: float) -> np.ndarray:
    sin_angle = np.sin(angle)
    cos_angle = np.cos(angle)

    rotation_matrix = np.array(
        [[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0, 0, 1]]
    )

    rotated_points = (rotation_matrix @ point_coordinates.T).T

    return rotated_points


def get_y_rotated_points(point_coordinates: np.ndarray, angle: float) -> np.ndarray:
    sin_angle = np.sin(angle)
    cos_angle = np.cos(angle)

    rotation_matrix = np.array(
        [[cos_angle, 0, sin_angle], [0, 1, 0], [-sin_angle, 0, cos_angle]]
    )

    rotated_points = (rotation_matrix @ point_coordinates.T).T

    return rotated_points


def get_ray_ellipsoid_intersection_points(
    ray_directions: np.ndarray,
    ellipsoid_radii: np.ndarray,
    ray_origins: np.ndarray | None = None,
    ellipsoid_center: np.ndarray | None = None,
    ellipsoid_orientation: np.ndarray | None = None,
    far_points: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    if ray_origins is None:
        ray_origins = np.zeros(3)

    if ellipsoid_center is None:
        ellipsoid_center = np.zeros(3)

    offset = (ray_origins - ellipsoid_center).reshape(-1, 3)
    directions = ray_directions.reshape(-1, 3)

    if ellipsoid_orientation is not None:
        transform_matrix = np.linalg.inv(ellipsoid_orientation)

        offset = (transform_matrix @ offset.T).T
        directions = (transform_matrix @ directions.T).T

    rneg2 = 1 / ellipsoid_radii**2
    d_rneg2 = directions * rneg2

    a = np.einsum("ij,ij->i", directions, d_rneg2)
    b = 2 * np.einsum("ij,ij->i", offset, d_rneg2)
    c = np.einsum("ij,j->i", offset**2, rneg2) - 1

    discriminant = b**2 - 4 * a * c
    discriminant[discriminant < 0] = np.nan

    t1 = -0.5 * b / a
    t2 = 0.5 * np.sqrt(discriminant) / a

    s1 = t1 + t2
    s2 = t1 - t2

    s1[s1 < 0] = np.inf
    s2[s2 < 0] = np.inf

    distance_condition = s2 > s1

    close_s = np.where(distance_condition, s1, s2)
    close_s[np.isinf(close_s)] = np.nan

    close_intersection_points = ray_origins + ray_directions * close_s.reshape(-1, 1)

    if far_points:
        far_s = np.where(~distance_condition, s1, s2)
        far_s[np.isinf(far_s)] = np.nan

        far_intersection_points = ray_origins + ray_directions * far_s.reshape(-1, 1)

        return close_intersection_points, far_intersection_points

    return close_intersection_points


def get_frames(
    pds3_image: PDS3Image, frame_height: int
) -> tuple[np.ndarray, np.ndarray]:
    image_data: np.ndarray = pds3_image.image  # type: ignore

    frames = einops.rearrange(
        image_data,
        "(num_frames frame_height) image_width -> num_frames frame_height image_width",
        frame_height=frame_height,
        image_width=image_data.shape[1],
    )

    pds3_label: PVLModule = pds3_image.label  # type: ignore

    start_time_utc: datetime = pds3_label["START_TIME"]  # type: ignore
    stop_time_utc: datetime = pds3_label["STOP_TIME"]  # type: ignore
    interframe_delay: Quantity = pds3_label["INTERFRAME_DELAY"]  # type: ignore

    start_time_et = spice.str2et(start_time_utc.strftime(r"%Y-%m-%d %H:%M:%S.%f"))
    stop_time_et = spice.str2et(stop_time_utc.strftime(r"%Y-%m-%d %H:%M:%S.%f"))
    interframe_delay_s = interframe_delay.value

    start_time_correction = spice.gdpool("INS-61500_START_TIME_BIAS", 0, 1)[0]
    interframe_delay_correction = spice.gdpool("INS-61500_INTERFRAME_DELTA", 0, 1)[0]

    start_time_et += start_time_correction
    stop_time_et += start_time_correction
    interframe_delay_s += interframe_delay_correction

    frame_times_et = np.arange(start_time_et, stop_time_et, interframe_delay_s)

    return frames, frame_times_et


def get_frame_framelets(frames: np.ndarray, framelet_height: int) -> list[np.ndarray]:
    return [get_framelets(frame, framelet_height) for frame in frames]


def get_frame_ellipsoid_positions_and_orientations(
    frame_times_et: np.ndarray,
    planet_name: str,
    spacecraft_name: str,
    planet_frame: str,
    camera_frame: str,
    abberation_correction: str = "LT+S",
) -> list[tuple[np.ndarray, np.ndarray]]:
    ellipsoid_positions_and_orientations = [
        get_body_position_and_orientation(
            planet_name,
            spacecraft_name,
            planet_frame,
            camera_frame,
            frame_time_et,
            abberation_correction=abberation_correction,
        )
        for frame_time_et in frame_times_et
    ]

    return ellipsoid_positions_and_orientations


def get_frame_spacecraft_positions(
    spacecraft_name: str,
    frame_times_et: np.ndarray,
    planet_name: str,
    planet_frame: str,
    abberation_correction: str = "LT+S",
) -> np.ndarray:
    spacecraft_positions = spice.spkpos(
        spacecraft_name,
        frame_times_et,
        planet_frame,
        abberation_correction,
        planet_name,
    )[0]

    return spacecraft_positions


def get_body_position_and_orientation(
    body_name: str,
    observer_name: str,
    body_frame: str,
    observer_frame: str,
    time_et: float,
    abberation_correction: str = "LT+S",
) -> tuple[np.ndarray, np.ndarray]:
    body_position = spice.spkpos(
        body_name, time_et, observer_frame, abberation_correction, observer_name
    )[0]
    body_orientation = spice.pxform(body_frame, observer_frame, time_et)

    return body_position, body_orientation


def find_corner_pixel_indices(
    patch_indices: np.ndarray, patch_resolution: int
) -> np.ndarray:
    corner_pixels = [
        [0, 0],
        [0, patch_resolution - 1],
        [patch_resolution - 1, 0],
        [patch_resolution - 1, patch_resolution - 1],
    ]

    corner_pixel_indices = np.where(
        np.all(np.isin(patch_indices, corner_pixels), axis=1)
    )[0]

    return corner_pixel_indices


def get_surface_point_solar_incidence_angles_deg(
    point_coordinates: np.ndarray,
    target_body: str,
    target_frame: str,
    time_et: float,
    surface_normal_method: str,
    abberation_correction: str = "LT+S",
) -> np.ndarray:
    point_normals = np.array(
        spice.srfnrm(
            surface_normal_method,
            target_body,
            time_et,
            target_frame,
            point_coordinates.tolist(),
        )
    )

    sun_position = spice.spkpos(
        "SUN", time_et, target_frame, abberation_correction, target_body
    )[0]

    sun_vectors = sun_position - point_coordinates

    point_incidence_angles = np.arccos(
        (sun_vectors * point_normals).sum(axis=1)
        / (np.linalg.norm(sun_vectors, axis=1) * np.linalg.norm(point_normals, axis=1))
    )

    return np.rad2deg(point_incidence_angles)


def get_framelets(frame: np.ndarray, framelet_height: int) -> np.ndarray:
    framelets = einops.rearrange(
        frame,
        "(num_filters framelet_height) frame_width "
        "-> num_filters framelet_height frame_width",
        framelet_height=framelet_height,
        frame_width=frame.shape[1],
    )

    return framelets


def junocam_distort(cam_coordinates: np.ndarray, k1: float, k2: float) -> np.ndarray:
    # See SPICE ik kernel "juno_junocam_v03.ti"

    r2 = (cam_coordinates**2).sum(axis=1)
    dr = 1 + k1 * r2 + k2 * r2**2
    cam_coordinates *= dr.reshape(-1, 1)

    return cam_coordinates


def junocam_to_framelet_coordinates(
    junocam_frame_points: np.ndarray, parameters: dict[str, float]
) -> np.ndarray:
    # See SPICE ik kernel "juno_junocam_v03.ti"

    cx, cy, k1, k2, fl = (parameters[param] for param in ["cx", "cy", "k1", "k2", "fl"])

    alpha = junocam_frame_points[:, 2] / fl
    cam_coordinates = junocam_frame_points[:, :2] / alpha.reshape(-1, 1)
    cam_coordinates = junocam_distort(cam_coordinates, k1, k2)
    framelet_coordinates = cam_coordinates + [cx, cy]

    return framelet_coordinates


def process_patch_image(
    patch_image: np.ndarray,
    patch_image_overlaps: np.ndarray,
    max_image_value: float,
    min_image_value: float,
) -> np.ndarray:
    patch_image_overlaps[patch_image_overlaps == 0] = 1
    patch_image /= patch_image_overlaps

    # Undo SQROOT sampling (see .LBL files) NOTE: not sure about this?
    # patch_image = patch_image**2
    # max_image_value = max_image_value**2
    # min_image_value = min_image_value**2

    patch_image = np.flip(patch_image, axis=2)  # BGR -> RGB

    image_range = max_image_value - min_image_value

    if image_range == 0:
        image_range = 1

    patch_image = ((patch_image - min_image_value) / image_range * 255).astype(np.uint8)

    return patch_image


def save_patch_image(patch_image: np.ndarray, file_path: Path) -> None:
    Image.fromarray(patch_image).save(file_path)
