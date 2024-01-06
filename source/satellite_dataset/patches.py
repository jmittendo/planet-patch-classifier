from pathlib import Path

import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.ma import MaskedArray
from pandas import DataFrame
from PIL import Image
from scipy import interpolate, ndimage, stats

import source.satellite_dataset.config as sdcfg
import source.satellite_dataset.table as sd_table
import source.satellite_dataset.utility as sd_util
import user.config as ucfg
from source.satellite_dataset.typing import (
    ImgGeoDataArrays,
    PatchCoordinate,
    PatchImageFormat,
    PatchInterpolation,
    PatchNormalization,
    PatchProjection,
    SatelliteDataArchive,
    SatelliteDataset,
    SphericalData,
)


class PatchGenerator:
    def __init__(
        self,
        patch_scale_km: float,
        patch_resolution: int,
        min_patch_density: float,
        num_density_bins: int,
        min_bin_density: float,
        interpolation_method: PatchInterpolation,
    ) -> None:
        self._patch_scale_km = patch_scale_km
        self._patch_resolution = patch_resolution
        self._min_patch_density = min_patch_density
        self._num_density_bins = num_density_bins
        self._min_bin_density = min_bin_density
        self._interpolation_method = interpolation_method

    def generate(
        self,
        spherical_data: SphericalData,
        output_file_path_base: Path,
        patch_img_format: PatchImageFormat,
        patch_normalization: PatchNormalization,
    ) -> None:
        half_patch_size = self._patch_scale_km / spherical_data["radius_km"]
        patch_coordinates = self._get_patch_coordinates(spherical_data, half_patch_size)
        patch_xy_range = (-half_patch_size, half_patch_size)
        patch_projections = self._get_patch_projections(
            spherical_data, patch_coordinates, patch_xy_range
        )
        patch_images = self._get_interpolated_patch_images(
            patch_projections, patch_xy_range
        )

        self._save_patch_images(
            patch_images,
            output_file_path_base,
            patch_img_format,
            patch_normalization,
            spherical_data,
        )

    def _get_patch_coordinates(
        self, spherical_data: SphericalData, half_patch_size: float
    ) -> list[PatchCoordinate]:
        phi_patches_angle = 2 * np.arcsin(0.5 * half_patch_size)

        # Offsets so that first and last phi angles are on the negative x-axis (because
        # their delta may not be equal to phi_patches_angle) and one angle is always on
        # (0, 0).
        phi_start = np.pi + np.pi % phi_patches_angle
        theta_start = (0.5 * np.pi) % phi_patches_angle

        phi_coords = np.arange(phi_start, 2 * np.pi + phi_start, phi_patches_angle) % (
            2 * np.pi
        )
        theta_coords = np.arange(theta_start, np.pi, phi_patches_angle)

        phi_grid, theta_grid = np.meshgrid(phi_coords, theta_coords)

        y_values = spherical_data["y_values"]
        z_values = spherical_data["z_values"]

        # Factor 1.01 because of float comparison problems
        range_y = (y_values.min() * 1.01, y_values.max() * 1.01)
        range_z = (z_values.min() * 1.01, z_values.max() * 1.01)

        patch_coords_phi = phi_grid.flatten()
        patch_coords_theta = theta_grid.flatten()

        sin_coords_phi = np.sin(patch_coords_phi)
        cos_coords_phi = np.cos(patch_coords_phi)
        sin_coords_theta = np.sin(patch_coords_theta)
        cos_coords_theta = np.cos(patch_coords_theta)

        patch_coords_x = cos_coords_phi * sin_coords_theta
        patch_coords_y = sin_coords_phi * sin_coords_theta
        patch_coords_z = cos_coords_theta

        invalid_coords_mask = (
            (patch_coords_x < 0)
            | (patch_coords_y > range_y[1])
            | (patch_coords_y < range_y[0])
            | (patch_coords_z > range_z[1])
            | (patch_coords_z < range_z[0])
        )

        patch_coords_phi = patch_coords_phi[~invalid_coords_mask]
        patch_coords_theta = patch_coords_theta[~invalid_coords_mask]

        patch_coordinates: list[PatchCoordinate] = [
            {"phi": phi, "theta": theta}
            for phi, theta in zip(patch_coords_phi, patch_coords_theta)
        ]

        return patch_coordinates

    def _get_patch_projections(
        self,
        spherical_data: SphericalData,
        patch_coordinates: list[PatchCoordinate],
        patch_xy_range: tuple[float, float],
    ) -> list[PatchProjection]:
        img_values = spherical_data["img_values"]

        projections: list[PatchProjection] = []

        for patch_coordinate in patch_coordinates:
            # Note that the projection will occur in x-direction and therefore the x-
            # and y-coordinates in the 2d projections are actually the y- and
            # z-coordinates of the spherical coordinates.
            rotated_sphere_points = self._get_rotated_sphere_points(
                spherical_data, patch_coordinate
            )
            visible_points_mask = self._get_visible_points_mask(
                rotated_sphere_points, patch_xy_range
            )
            patch_points = rotated_sphere_points[1:, visible_points_mask]
            num_points = patch_points.shape[1]

            # Checks to make sure the projection segment is suitable for interpolation
            if not self._passes_global_density_check(num_points):
                continue

            if not self._passes_local_density_check(patch_points, patch_xy_range):
                continue

            patch_img_values = img_values[visible_points_mask]

            projection: PatchProjection = {
                "img_values": patch_img_values,
                "x_values": patch_points[0],
                "y_values": patch_points[1],
            }

            projections.append(projection)

        return projections

    def _get_rotated_sphere_points(
        self, spherical_data: SphericalData, patch_coordinate: PatchCoordinate
    ) -> ndarray:
        x_values = spherical_data["x_values"]
        y_values = spherical_data["y_values"]
        z_values = spherical_data["z_values"]
        xyz_array = np.row_stack([x_values, y_values, z_values])

        z_rot_angle = 0.5 * np.pi - patch_coordinate["theta"]
        y_rot_angle = -patch_coordinate["phi"]
        rotation_matrix = sd_util.get_zy_rotation_matrix(z_rot_angle, y_rot_angle)

        return np.dot(rotation_matrix, xyz_array)

    def _get_visible_points_mask(
        self, sphere_points: ndarray, patch_xy_range: tuple[float, float]
    ) -> ndarray:
        visible_points_mask_x = sphere_points[0] >= 0

        # Factor 1.5 for selecting points outside the final projection area that are
        # necessary for the interpolation at the borders
        visible_points_mask_y_1 = sphere_points[1] >= 1.5 * patch_xy_range[0]
        visible_points_mask_y_2 = sphere_points[1] <= 1.5 * patch_xy_range[1]

        visible_points_mask_z_1 = sphere_points[2] >= 1.5 * patch_xy_range[0]
        visible_points_mask_z_2 = sphere_points[2] <= 1.5 * patch_xy_range[1]

        visible_points_mask = (
            visible_points_mask_x
            & visible_points_mask_y_1
            & visible_points_mask_y_2
            & visible_points_mask_z_1
            & visible_points_mask_z_2
        )

        return visible_points_mask

    def _passes_global_density_check(self, num_points: int) -> bool:
        # Factor 2.25 because of factor 1.5 in _get_visible_points_mask
        passes_check = (
            num_points / 2.25 >= self._min_patch_density * self._patch_resolution**2
        )

        return passes_check

    def _passes_local_density_check(
        self, patch_points: np.ndarray, patch_xy_range: tuple[float, float]
    ) -> bool:
        range_array = np.asarray([patch_xy_range, patch_xy_range])

        bin_densities, _, _ = np.histogram2d(
            patch_points[0],
            patch_points[1],
            bins=self._num_density_bins,
            range=range_array,
        )

        min_density = (
            self._min_bin_density
            * (self._patch_resolution / self._num_density_bins) ** 2
        )

        return not (bin_densities < min_density).any()

    def _get_interpolated_patch_images(
        self,
        patch_projections: list[PatchProjection],
        patch_xy_range: tuple[float, float],
    ) -> list[ndarray]:
        pixel_xy_values = np.linspace(*patch_xy_range, self._patch_resolution)
        pixel_x_grid, pixel_y_grid = np.meshgrid(pixel_xy_values, pixel_xy_values)

        patch_images: list[ndarray] = []

        for projection in patch_projections:
            projection_img_values = projection["img_values"]
            projection_x_values = projection["x_values"]
            projection_y_values = projection["y_values"]

            projection_xy_values = np.column_stack(
                [projection_x_values, projection_y_values]
            )

            interpolated_projection = interpolate.griddata(
                projection_xy_values,
                projection_img_values,
                (pixel_x_grid, pixel_y_grid),
                method=self._interpolation_method,
            )

            if np.isnan(interpolated_projection).any():
                continue

            patch_img = np.flip(interpolated_projection, axis=0)

            patch_images.append(patch_img)

        return patch_images

    def _save_patch_images(
        self,
        patch_images: list[ndarray],
        output_file_path_base: Path,
        patch_img_format: PatchImageFormat,
        patch_normalization: PatchNormalization,
        full_spherical_data: SphericalData,
    ):
        output_file_path_base.parent.mkdir(parents=True, exist_ok=True)

        for i, patch_img in enumerate(patch_images):
            normalized_patch_images = []
            normalization_modes = []

            if patch_normalization not in ("local", "global", "both"):
                raise ValueError(
                    f"{patch_normalization} is not a valid patch normalization mode"
                )

            if patch_normalization in ("local", "both"):
                normalized_patch_img = sd_util.get_normalized_img(patch_img)

                normalized_patch_images.append(normalized_patch_img)
                normalization_modes.append("local")

            if patch_normalization in ("global", "both"):
                full_img_values = full_spherical_data["img_values"]
                global_min = full_img_values.min()
                global_max = full_img_values.max()

                normalized_patch_img = (patch_img - global_min) / (
                    global_max - global_min
                )

                normalized_patch_images.append(normalized_patch_img)
                normalization_modes.append("global")

            for patch_img, normalization_mode in zip(
                normalized_patch_images, normalization_modes
            ):
                normalization_str = "ln" if normalization_mode == "local" else "gn"

                output_file_name_base = output_file_path_base.name
                output_file_name = (
                    f"{output_file_name_base}_patch_{i}_{normalization_str}"
                    f".{patch_img_format}"
                )
                output_file_path = output_file_path_base.with_name(output_file_name)

                match patch_img_format:
                    case "png" | "jpg":
                        int_image_array = (patch_img * 255).astype(np.uint8)
                        Image.fromarray(int_image_array).save(output_file_path)
                    case "txt":
                        np.savetxt(output_file_path, patch_img)
                    case "npy":
                        np.save(output_file_path, patch_img)
                    case _:
                        raise ValueError(
                            f"'{patch_img_format}' is not a valid patch image format"
                        )


def generate_patches(
    dataset: SatelliteDataset,
    patch_scale_km: float,
    patch_resolution: int,
    patch_normalization: PatchNormalization,
    regenerate_table: bool,
) -> None:
    dataset_archive = sd_util.load_archive(dataset["archive"])
    table_path = sdcfg.DATASET_TABLES_DIR_PATH / f"{dataset['name']}.pkl"

    if regenerate_table or not table_path.is_file():
        sd_table.generate_dataset_table(dataset, dataset_archive, table_path)

    dataset_table: DataFrame = pd.read_pickle(table_path)

    output_dir_path = sdcfg.PATCHES_DIR_PATH / dataset["name"]
    output_dir_path.mkdir(parents=True, exist_ok=True)

    archive_type = dataset_archive["type"]

    match archive_type:
        case "img-geo":
            _generate_img_geo_patches(
                dataset_archive,
                dataset_table,
                patch_scale_km,
                patch_resolution,
                patch_normalization,
                output_dir_path,
            )
        case "img-spice":
            raise NotImplementedError
        case _:
            raise ValueError(
                "No patch generation script implemented for archive type "
                f"'{archive_type}'"
            )


def _generate_img_geo_patches(
    archive: SatelliteDataArchive,
    dataset_table: DataFrame,
    patch_scale_km: float,
    patch_resolution: int,
    patch_normalization: PatchNormalization,
    output_dir_path: Path,
) -> None:
    # Spatial resolution of a patch in m/px (ignoring projection effects / distortions)
    patch_resolution_mpx = patch_scale_km * 1000 / patch_resolution

    for row_index, row_data in dataset_table.iterrows():
        img_max_resolution_mpx: float = row_data["max_resolution_mpx"]

        if not _passes_resolution_threshold(
            img_max_resolution_mpx, patch_resolution_mpx
        ):
            continue

        img_file_path = Path(row_data["img_file_path"])
        geo_file_path = Path(row_data["geo_file_path"])

        data_arrays = _load_img_geo_data_arrays(archive, img_file_path, geo_file_path)

        _apply_invalid_mask(archive, data_arrays)
        _apply_background_mask(data_arrays, ucfg.PATCH_BACKGROUND_ANGLE)
        _normalize_img_intensity(data_arrays)
        _apply_outlier_mask(data_arrays, ucfg.PATCH_OUTLIER_SIGMA)

        img_array = data_arrays["image"]
        lon_array = data_arrays["longitude"]
        lat_array = data_arrays["latitude"]

        img_values = img_array[~img_array.mask]
        lon_values = lon_array[~lon_array.mask]
        lat_values = lat_array[~lat_array.mask]

        spherical_data = _get_spherical_data(
            img_values, lon_values, lat_values, archive["planet_radius_km"]
        )

        output_file_path_base = output_dir_path / row_data["file_name_base"]

        patch_generator = PatchGenerator(
            patch_scale_km,
            patch_resolution,
            ucfg.MIN_PATCH_DENSITY,
            ucfg.NUM_PATCH_DENSITY_BINS,
            ucfg.MIN_PATCH_BIN_DENSITY,
            ucfg.PATCH_INTERPOLATION_METHOD,
        )
        patch_generator.generate(
            spherical_data,
            output_file_path_base,
            ucfg.PATCH_IMAGE_FORMAT,
            patch_normalization,
        )


def _load_img_geo_data_arrays(
    archive: SatelliteDataArchive, img_file_path: Path, geo_file_path: Path
) -> ImgGeoDataArrays:
    archive_name = archive["name"]

    match archive_name:
        case "vex-vmc":
            img_array = sd_util.load_pds3_data(img_file_path)[0]
            (
                ina_array,  # Incidence angle data
                ema_array,  # Emission angle data
                pha_array,  # Phase angle data (not needed)
                lat_array,  # Latitude data
                lon_array,  # Longitude data
            ) = sd_util.load_pds3_data(geo_file_path)
        case "vco":
            img_array = sd_util.load_fits_data(img_file_path, 1)
            (
                ina_array,  # Incidence angle data
                ema_array,  # Emission angle data
                lat_array,  # Latitude data
                lon_array,  # Longitude data
            ) = sd_util.load_fits_data(
                geo_file_path,
                [
                    "Incidence angle",
                    "Emission angle",
                    "Latitude",
                    "Longitude",
                ],
            )

            lon_array = sd_util.fix_360_longitude(lon_array)
        case _:
            raise ValueError(
                f"Can not load data arrays for unknown archive {archive_name}"
            )

    data_arrays: ImgGeoDataArrays = {
        "image": MaskedArray(data=img_array.astype(float)),
        "incidence_angle": MaskedArray(data=ina_array.astype(float)),
        "emission_angle": MaskedArray(data=ema_array.astype(float)),
        "latitude": MaskedArray(data=lat_array.astype(float)),
        "longitude": MaskedArray(data=lon_array.astype(float)),
    }

    return data_arrays


def _apply_invalid_mask(
    archive: SatelliteDataArchive, data_arrays: ImgGeoDataArrays
) -> None:
    archive_name = archive["name"]

    match archive_name:
        case "vex-vmc":
            # The invalidity comparison values could be done more "exact" but this
            # should work
            invalid_mask = data_arrays["image"] < 0

            array: MaskedArray
            for array_name, array in data_arrays.items():  # type: ignore
                if array_name == "image":
                    continue

                invalid_mask |= array < -1e10
        case "vco":
            # The invalidity comparison values could be done more "exact" but this
            # should work
            invalid_mask = data_arrays["image"] < -1e38

            array: MaskedArray
            for array_name, array in data_arrays.items():  # type: ignore
                if array_name == "image":
                    continue

                invalid_mask |= ~np.isfinite(array)
        case _:
            raise ValueError(
                f"No invalid-mask code implemented for archive '{archive_name}'"
            )

    _apply_img_geo_arrays_mask(data_arrays, invalid_mask)


def _apply_background_mask(
    data_arrays: ImgGeoDataArrays, threshold_angle_deg: float
) -> None:
    unilluminated_mask = data_arrays["incidence_angle"] > threshold_angle_deg
    observation_mask = data_arrays["emission_angle"] > threshold_angle_deg
    background_mask = unilluminated_mask | observation_mask

    _apply_img_geo_arrays_mask(data_arrays, background_mask)


def _apply_outlier_mask(data_arrays: ImgGeoDataArrays, sigma_threshold: float) -> None:
    img_array = data_arrays["image"]

    filtered_img_array = ndimage.median_filter(img_array, size=3)
    diff_array = filtered_img_array - img_array
    outlier_mask = np.abs(diff_array) > sigma_threshold * diff_array.std()

    _apply_img_geo_arrays_mask(data_arrays, outlier_mask)


def _apply_img_geo_arrays_mask(data_arrays: ImgGeoDataArrays, mask: ndarray) -> None:
    array: MaskedArray
    for array in data_arrays.values():  # type: ignore
        array.mask |= mask


def _normalize_img_intensity(data_arrays: ImgGeoDataArrays) -> None:
    # Normalization using Minnaert's law

    img_array = data_arrays["image"]
    ina_array = data_arrays["incidence_angle"]
    ema_array = data_arrays["emission_angle"]

    valid_mask = img_array.mask

    if not valid_mask.any():
        return

    cos_ina_array = np.cos(np.deg2rad(ina_array))
    cos_ema_array = np.cos(np.deg2rad(ema_array))

    img_values = img_array[valid_mask]
    cos_ina_values = cos_ina_array[valid_mask]
    cos_ema_values = cos_ema_array[valid_mask]

    linreg_x_arg = cos_ina_values * cos_ema_values
    linreg_y_arg = img_values * cos_ema_values

    positive_mask = (linreg_x_arg > 0) & (linreg_y_arg > 0)

    if not positive_mask.any():
        return

    linreg_x = np.log(linreg_x_arg[positive_mask])
    linreg_y = np.log(linreg_y_arg[positive_mask])

    if not np.any(linreg_x != linreg_x[0]):
        return

    linreg_result = stats.linregress(linreg_x, linreg_y)
    slope = linreg_result.slope  # type: ignore

    if np.isnan(slope):
        return

    img_array[...] = img_array / (cos_ema_array ** (slope - 1) * cos_ina_array**slope)


def _passes_resolution_threshold(
    img_max_resolution: float, patch_resolution: float
) -> bool:
    return img_max_resolution / patch_resolution < ucfg.PATCH_RESOLUTION_TOLERANCE


def _get_spherical_data(
    img_values: ndarray,
    longitude_values: ndarray,
    latitude_values: ndarray,
    planet_radius_km: float,
) -> SphericalData:
    phi_values = np.deg2rad(longitude_values % 360)
    theta_values = np.deg2rad(90 - latitude_values)

    sin_phi_values = np.sin(phi_values)
    cos_phi_values = np.cos(phi_values)
    sin_theta_values = np.sin(theta_values)
    cos_theta_values = np.cos(theta_values)

    x_values = sin_theta_values * cos_phi_values
    y_values = sin_theta_values * sin_phi_values
    z_values = cos_theta_values

    # Find the approximate center of the points on the sphere and shift it to phi = 0
    # and theta = 0 as this should lead to better spacing between the patches
    xyz_array = np.row_stack([x_values, y_values, z_values])
    xyz_median = np.median(xyz_array, axis=1)

    median_x, median_y, median_z = xyz_median / np.linalg.norm(xyz_median)

    median_phi = np.sign(median_y) * np.arccos(
        median_x / np.sqrt(median_x**2 + median_y**2)
    )
    median_theta = np.arccos(median_z)

    y_rotation_angle = np.pi / 2 - median_theta
    z_rotation_angle = -median_phi

    rotation_matrix = sd_util.get_zy_rotation_matrix(z_rotation_angle, y_rotation_angle)
    xyz_array = np.dot(rotation_matrix, xyz_array)

    spherical_data_values: SphericalData = {
        "img_values": img_values,
        "x_values": xyz_array[0],
        "y_values": xyz_array[1],
        "z_values": xyz_array[2],
        "radius_km": planet_radius_km,
    }

    return spherical_data_values
