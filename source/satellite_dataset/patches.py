from pathlib import Path

import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.ma import MaskedArray
from pandas import DataFrame
from scipy import ndimage, stats

import source.satellite_dataset.config as sdcfg
import source.satellite_dataset.table as sd_table
import source.satellite_dataset.utility as sd_util
import user.config as ucfg
from source.satellite_dataset.typing import (
    ImgGeoDataArrays,
    SatelliteDataArchive,
    SatelliteDataset,
    SphericalDataValues,
)


def generate_patches(
    dataset: SatelliteDataset,
    patch_scale_km: float,
    patch_resolution: int,
    regenerate_table: bool,
) -> None:
    dataset_archive = sd_util.load_archive(dataset["archive"])
    table_path = sdcfg.DATASET_TABLES_DIR_PATH / f"{dataset['name']}.pkl"

    if regenerate_table or not table_path.is_file():
        sd_table.generate_dataset_table(dataset, dataset_archive, table_path)

    dataset_table: DataFrame = pd.read_pickle(table_path)

    # Spatial resolution of a patch in m/px (ignoring projection effects / distortions)
    patch_resolution_mpx = patch_scale_km * 1000 / patch_resolution

    archive_type = dataset_archive["type"]

    match archive_type:
        case "img-geo":
            _generate_img_geo_patches(
                dataset_archive, dataset_table, patch_resolution_mpx
            )
        case "img-spice":
            raise NotImplementedError
        case _:
            raise ValueError(
                "No patch generation script implemented for archive type "
                f"'{archive_type}'"
            )


def _generate_img_geo_patches(
    archive: SatelliteDataArchive, dataset_table: DataFrame, patch_resolution_mpx: float
) -> None:
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

        spherical_data = _get_spherical_data(img_values, lon_values, lat_values)


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
    img_values: ndarray, longitude_values: ndarray, latitude_values: ndarray
) -> SphericalDataValues:
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

    spherical_data_values: SphericalDataValues = {
        "image": img_values,
        "x": xyz_array[0],
        "y": xyz_array[1],
        "z": xyz_array[2],
    }

    return spherical_data_values
