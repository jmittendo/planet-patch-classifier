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
from pathlib import Path

import numpy as np
from numpy import ndarray
from numpy.ma import MaskedArray
from PIL import Image
from scipy import interpolate, ndimage, stats

import source.satellite_dataset.utility as sd_util
import source.utility as util
from source.satellite_dataset.patches.typing import (
    ImgGeoDataArrays,
    ImgGeoPatchInterpolation,
    ImgGeoPatchProjection,
    PatchCoordinate,
    SphericalData,
)

if typing.TYPE_CHECKING:
    from source.satellite_dataset.archive import Archive


class ImgGeoPatchGenerator:
    def __init__(
        self,
        patch_scale_km: float,
        patch_resolution: int,
        min_patch_density: float,
        num_density_bins: int,
        min_bin_density: float,
        interpolation_method: ImgGeoPatchInterpolation,
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
        output_dir_path: Path,
        output_file_name_base: str,
        global_normalization: bool = False,
    ) -> tuple[list[str], list[PatchCoordinate]]:
        spherical_data, back_rotation_matrix = self._center_spherical_data(
            spherical_data
        )
        half_patch_size = 0.5 * self._patch_scale_km / spherical_data["radius_km"]
        patch_coordinates = self._get_patch_coordinates(
            spherical_data, half_patch_size, back_rotation_matrix
        )
        patch_xy_range = (-half_patch_size, half_patch_size)
        patch_projections, projection_coordinates = self._get_patch_projections(
            spherical_data, patch_coordinates, patch_xy_range
        )
        patch_images, img_coordinates = self._get_interpolated_patch_images(
            patch_projections, projection_coordinates, patch_xy_range
        )

        img_file_names = self._save_patch_images(
            patch_images,
            output_dir_path,
            output_file_name_base,
            global_normalization,
            spherical_data,
        )

        return img_file_names, img_coordinates

    def _center_spherical_data(
        self, spherical_data: SphericalData
    ) -> tuple[SphericalData, ndarray]:
        # Find the approximate center of the points on the sphere and shift it to
        # phi = 0 and theta = 0 as this should lead to better spacing between the
        # patches
        x_values = spherical_data["x_values"]
        y_values = spherical_data["y_values"]
        z_values = spherical_data["z_values"]

        xyz_array = np.row_stack([x_values, y_values, z_values])
        xyz_median = np.median(xyz_array, axis=1)

        median_x, median_y, median_z = xyz_median / np.linalg.norm(xyz_median)

        median_phi = np.sign(median_y) * np.arccos(
            median_x / np.sqrt(median_x**2 + median_y**2)
        )
        median_theta = np.arccos(median_z)

        y_rotation_angle = np.pi / 2 - median_theta
        z_rotation_angle = -median_phi

        rotation_matrix = sd_util.get_zy_rotation_matrix(
            z_rotation_angle, y_rotation_angle
        )
        xyz_array = np.dot(rotation_matrix, xyz_array)

        centered_spherical_data: SphericalData = {
            "img_values": spherical_data["img_values"],
            "x_values": xyz_array[0],
            "y_values": xyz_array[1],
            "z_values": xyz_array[2],
            "radius_km": spherical_data["radius_km"],
            "solar_longitude": spherical_data["solar_longitude"],
        }

        return centered_spherical_data, np.linalg.inv(rotation_matrix)

    def _get_patch_coordinates(
        self,
        spherical_data: SphericalData,
        half_patch_size: float,
        back_rotation_matrix: ndarray,
    ) -> list[PatchCoordinate]:
        y_values = spherical_data["y_values"]
        z_values = spherical_data["z_values"]

        if y_values.size == 0:
            return []

        phi_patches_angle = 2 * np.arcsin(half_patch_size)

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

        patch_coords_phi = phi_grid.flatten()
        patch_coords_theta = theta_grid.flatten()

        sin_coords_phi = np.sin(patch_coords_phi)
        cos_coords_phi = np.cos(patch_coords_phi)
        sin_coords_theta = np.sin(patch_coords_theta)
        cos_coords_theta = np.cos(patch_coords_theta)

        patch_coords_x = cos_coords_phi * sin_coords_theta
        patch_coords_y = sin_coords_phi * sin_coords_theta
        patch_coords_z = cos_coords_theta

        # Factor 1.01 because of float comparison problems
        range_y = (y_values.min() * 1.01, y_values.max() * 1.01)
        range_z = (z_values.min() * 1.01, z_values.max() * 1.01)

        invalid_coords_mask = (
            (patch_coords_x < 0)
            | (patch_coords_y > range_y[1])
            | (patch_coords_y < range_y[0])
            | (patch_coords_z > range_z[1])
            | (patch_coords_z < range_z[0])
        )

        patch_coords_x = patch_coords_x[~invalid_coords_mask]
        patch_coords_y = patch_coords_y[~invalid_coords_mask]
        patch_coords_z = patch_coords_z[~invalid_coords_mask]
        patch_coords_phi = patch_coords_phi[~invalid_coords_mask]
        patch_coords_theta = patch_coords_theta[~invalid_coords_mask]

        # Use back rotation matrix from spherical data centering to get original patch
        # coordinates
        solar_longitude = spherical_data["solar_longitude"]

        original_coords_x, original_coords_y, original_coords_z = np.dot(
            back_rotation_matrix,
            np.row_stack([patch_coords_x, patch_coords_y, patch_coords_z]),
        )

        original_coords_phi = np.arctan2(original_coords_y, original_coords_x) % (
            2 * np.pi
        )
        original_coords_theta = np.arccos(original_coords_z) % np.pi

        original_longitudes = sd_util.fix_360_longitude(np.rad2deg(original_coords_phi))
        original_latitudes = 90 - np.rad2deg(original_coords_theta)
        original_local_times = sd_util.longitude_to_local_time(
            original_longitudes, solar_longitude
        )

        patch_coordinates: list[PatchCoordinate] = [
            {
                "phi": phi,
                "theta": theta,
                "longitude": lon,
                "latitude": lat,
                "local_time": lt,
            }
            for phi, theta, lon, lat, lt, in zip(
                patch_coords_phi,
                patch_coords_theta,
                original_longitudes,
                original_latitudes,
                original_local_times,
            )
        ]

        return patch_coordinates

    def _get_patch_projections(
        self,
        spherical_data: SphericalData,
        patch_coordinates: list[PatchCoordinate],
        patch_xy_range: tuple[float, float],
    ) -> tuple[list[ImgGeoPatchProjection], list[PatchCoordinate]]:
        projections: list[ImgGeoPatchProjection] = []
        projection_coordinates: list[PatchCoordinate] = []

        for patch_coordinate in patch_coordinates:
            # Note that the projection will occur in x-direction and therefore the x-
            # and y-coordinates in the 2d projections are actually the y- and
            # z-coordinates of the spherical coordinates.
            rotated_spherical_data = self._get_rotated_sphere_points(
                spherical_data, patch_coordinate
            )
            visible_points_mask = self._get_visible_points_mask(
                rotated_spherical_data, patch_xy_range
            )
            num_points = np.count_nonzero(visible_points_mask)

            # Checks to make sure the projection segment is suitable for interpolation
            if not self._passes_global_density_check(num_points):
                continue

            patch_points = np.row_stack(
                [
                    rotated_spherical_data["y_values"][visible_points_mask],
                    rotated_spherical_data["z_values"][visible_points_mask],
                ]
            )

            if not self._passes_local_density_check(patch_points, patch_xy_range):
                continue

            patch_img_values = rotated_spherical_data["img_values"][visible_points_mask]

            projection: ImgGeoPatchProjection = {
                "img_values": patch_img_values,
                "x_values": patch_points[0],
                "y_values": patch_points[1],
            }

            projections.append(projection)
            projection_coordinates.append(patch_coordinate)

        return projections, projection_coordinates

    def _get_rotated_sphere_points(
        self, spherical_data: SphericalData, patch_coordinate: PatchCoordinate
    ) -> SphericalData:
        x_values = spherical_data["x_values"]
        y_values = spherical_data["y_values"]
        z_values = spherical_data["z_values"]
        xyz_array = np.row_stack([x_values, y_values, z_values])

        z_rot_angle = -patch_coordinate["phi"]
        y_rot_angle = 0.5 * np.pi - patch_coordinate["theta"]
        rotation_matrix = sd_util.get_zy_rotation_matrix(z_rot_angle, y_rot_angle)

        rotated_points = np.dot(rotation_matrix, xyz_array)

        rotated_spherical_data: SphericalData = {
            "img_values": spherical_data["img_values"],
            "x_values": rotated_points[0],
            "y_values": rotated_points[1],
            "z_values": rotated_points[2],
            "radius_km": spherical_data["radius_km"],
            "solar_longitude": spherical_data["solar_longitude"],
        }

        return rotated_spherical_data

    def _get_visible_points_mask(
        self, spherical_data: SphericalData, patch_xy_range: tuple[float, float]
    ) -> ndarray:
        visible_points_mask_x = spherical_data["x_values"] >= 0

        # Factor 1.5 for selecting points outside the final projection area that are
        # necessary for the interpolation at the borders
        visible_points_mask_y_1 = spherical_data["y_values"] >= 1.5 * patch_xy_range[0]
        visible_points_mask_y_2 = spherical_data["y_values"] <= 1.5 * patch_xy_range[1]

        visible_points_mask_z_1 = spherical_data["z_values"] >= 1.5 * patch_xy_range[0]
        visible_points_mask_z_2 = spherical_data["z_values"] <= 1.5 * patch_xy_range[1]

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
        patch_projections: list[ImgGeoPatchProjection],
        projection_coordinates: list[PatchCoordinate],
        patch_xy_range: tuple[float, float],
    ) -> tuple[list[ndarray], list[PatchCoordinate]]:
        pixel_xy_values = np.linspace(*patch_xy_range, self._patch_resolution)
        pixel_x_grid, pixel_y_grid = np.meshgrid(pixel_xy_values, pixel_xy_values)

        patch_images: list[ndarray] = []
        img_coordinates: list[PatchCoordinate] = []

        for projection_coordinate, projection in zip(
            projection_coordinates, patch_projections
        ):
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
            img_coordinates.append(projection_coordinate)

        return patch_images, img_coordinates

    def _save_patch_images(
        self,
        patch_images: list[ndarray],
        output_dir_path: Path,
        file_name_base: str,
        global_normalization: bool,
        full_spherical_data: SphericalData,
    ) -> list[str]:
        dataset_versions_dir_path = output_dir_path / "versions"
        dataset_versions_dir_path.mkdir(parents=True, exist_ok=True)

        img_file_names: list[str] = []

        for i, patch_img in enumerate(patch_images):
            normalized_patch_images = []
            normalization_modes = []

            normalized_patch_img = util.get_normalized_img(patch_img)

            normalized_patch_images.append(normalized_patch_img)
            normalization_modes.append("local")

            if global_normalization:
                full_img_values = full_spherical_data["img_values"]
                global_min = full_img_values.min()
                global_max = full_img_values.max()

                normalized_patch_img = (patch_img - global_min) / (
                    global_max - global_min
                )

                normalized_patch_images.append(normalized_patch_img)
                normalization_modes.append("global")

            output_file_name = f"{file_name_base}_patch-{i}.png"

            for patch_img, normalization_mode in zip(
                normalized_patch_images, normalization_modes
            ):
                version_dir_name = f"norm-{normalization_mode}"

                version_dir_path = dataset_versions_dir_path / version_dir_name
                version_dir_path.mkdir(parents=True, exist_ok=True)

                output_file_path = version_dir_path / output_file_name
                int_image_array = (patch_img * 255).astype(np.uint8)

                Image.fromarray(int_image_array).save(output_file_path)

            img_file_names.append(output_file_name)

        return img_file_names


def apply_invalid_mask(archive: "Archive", data_arrays: ImgGeoDataArrays) -> None:
    match archive.name:
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
                f"No invalid-mask code implemented for archive '{archive.name}'"
            )

    _apply_arrays_mask(data_arrays, invalid_mask)


def apply_angle_mask(data_arrays: ImgGeoDataArrays, threshold_angle_deg: float) -> None:
    unilluminated_mask = data_arrays["incidence_angle"] > threshold_angle_deg
    observation_mask = data_arrays["emission_angle"] > threshold_angle_deg
    angle_mask = unilluminated_mask | observation_mask

    _apply_arrays_mask(data_arrays, angle_mask)


def apply_outlier_mask(data_arrays: ImgGeoDataArrays, sigma_threshold: float) -> None:
    img_array = data_arrays["image"]

    filtered_img_array = ndimage.median_filter(img_array, size=3)
    diff_array = filtered_img_array - img_array
    outlier_mask = np.abs(diff_array) > sigma_threshold * diff_array.std()

    _apply_arrays_mask(data_arrays, outlier_mask)


def normalize_intensity(data_arrays: ImgGeoDataArrays) -> None:
    # Normalization using Minnaert's law

    img_array = data_arrays["image"]
    ina_array = data_arrays["incidence_angle"]
    ema_array = data_arrays["emission_angle"]

    valid_mask = ~img_array.mask

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


def get_spherical_data(
    img_values: ndarray,
    longitude_values: ndarray,
    latitude_values: ndarray,
    planet_radius_km: float,
    solar_longitude: float,
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

    spherical_data_values: SphericalData = {
        "img_values": img_values,
        "x_values": x_values,
        "y_values": y_values,
        "z_values": z_values,
        "radius_km": planet_radius_km,
        "solar_longitude": solar_longitude,
    }

    return spherical_data_values


def _apply_arrays_mask(data_arrays: ImgGeoDataArrays, mask: ndarray) -> None:
    array: MaskedArray
    for array in data_arrays.values():  # type: ignore
        array.mask |= mask
