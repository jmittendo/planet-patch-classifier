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

import json
import typing
from pathlib import Path

from numpy.ma import MaskedArray
from pandas import DataFrame
from tqdm import tqdm

import source.satellite_dataset.patches.img_geo_patches as sd_img_geo_patches
import source.satellite_dataset.patches.utility as sd_patch_util
import source.satellite_dataset.utility as sd_util
import user.config as user_config
from source import config
from source.satellite_dataset.patches.img_geo_patches import ImgGeoPatchGenerator
from source.satellite_dataset.patches.jno_jnc_patches import JnoJncPatchGenerator
from source.satellite_dataset.patches.typing import ImgGeoDataArrays

if typing.TYPE_CHECKING:
    from source.satellite_dataset import SatelliteDataset
    from source.satellite_dataset.archive import ImgGeoArchive, JnoJncArchive


def generate_img_geo_patches(
    archive: "ImgGeoArchive",
    dataset: "SatelliteDataset",
    patch_scale_km: float,
    patch_resolution: int,
    global_normalization: bool = False,
) -> None:
    output_dir_path = get_patch_dataset_path(
        dataset.name, patch_scale_km, patch_resolution
    )
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Spatial resolution of a patch in m/px (ignoring projection effects / distortions)
    patch_resolution_mpx = patch_scale_km * 1000 / patch_resolution

    patch_file_names: list[str] = []
    patch_longitudes: list[float] = []
    patch_latitudes: list[float] = []
    patch_local_times: list[float] = []

    for data in tqdm(dataset, desc="Generating patches"):
        img_max_resolution_mpx: float = data["max_resolution_mpx"]

        if not sd_patch_util.passes_resolution_threshold(
            img_max_resolution_mpx, patch_resolution_mpx
        ):
            continue

        img_file_path = Path(data["img_file_path"])
        geo_file_path = Path(data["geo_file_path"])

        data_arrays = archive.load_data_arrays(img_file_path, geo_file_path)

        sd_img_geo_patches.apply_invalid_mask(archive, data_arrays)
        sd_img_geo_patches.apply_angle_mask(
            data_arrays, user_config.PATCH_ANGLE_THRESHOLD
        )
        sd_img_geo_patches.normalize_intensity(data_arrays)
        sd_img_geo_patches.apply_outlier_mask(
            data_arrays, user_config.PATCH_OUTLIER_SIGMA
        )

        img_values = data_arrays["image"].compressed()
        lon_values = data_arrays["longitude"].compressed()
        lat_values = data_arrays["latitude"].compressed()

        if img_values.size == 0:
            continue

        solar_longitude = data["solar_longitude_deg"]

        spherical_data = sd_img_geo_patches.get_spherical_data(
            img_values,
            lon_values,
            lat_values,
            archive.planet.radius_km,
            solar_longitude,
        )

        patch_generator = ImgGeoPatchGenerator(
            patch_scale_km,
            patch_resolution,
            user_config.MIN_PATCH_DENSITY,
            user_config.NUM_PATCH_DENSITY_BINS,
            user_config.MIN_PATCH_BIN_DENSITY,
            user_config.PATCH_INTERPOLATION_METHOD,
        )
        img_file_names, patch_coordinates = patch_generator.generate(
            spherical_data,
            output_dir_path,
            data["file_name_base"],
            global_normalization=global_normalization,
        )

        patch_file_names += img_file_names

        for patch_coordinate in patch_coordinates:
            patch_longitudes.append(patch_coordinate["longitude"])
            patch_latitudes.append(patch_coordinate["latitude"])
            patch_local_times.append(patch_coordinate["local_time"])

    patch_info_table_dict = {
        "file_name": patch_file_names,
        "longitude": patch_longitudes,
        "latitude": patch_latitudes,
        "local_time": patch_local_times,
    }

    patch_info_table = DataFrame(data=patch_info_table_dict)
    patch_info_table.to_pickle(output_dir_path / "table.pkl")

    patch_dataset_info_dict = {
        "scale_km": patch_scale_km,
        "resolution": patch_resolution,
        "labels": [],
    }
    patch_dataset_info_json_path = output_dir_path / "info.json"

    with open(patch_dataset_info_json_path, "w") as patch_dataset_info_json:
        json.dump(
            patch_dataset_info_dict,
            patch_dataset_info_json,
            indent=user_config.JSON_INDENT,
        )


def load_vex_vmc_data_arrays(
    img_file_path: Path, geo_file_path: Path
) -> ImgGeoDataArrays:
    img_array = sd_util.load_pds3_data(img_file_path)[0]
    (
        ina_array,  # Incidence angle data
        ema_array,  # Emission angle data
        pha_array,  # Phase angle data (not needed)
        lat_array,  # Latitude data
        lon_array,  # Longitude data
    ) = sd_util.load_pds3_data(geo_file_path)

    data_arrays: ImgGeoDataArrays = {
        "image": MaskedArray(data=img_array.astype(float)),
        "incidence_angle": MaskedArray(data=ina_array.astype(float)),
        "emission_angle": MaskedArray(data=ema_array.astype(float)),
        "latitude": MaskedArray(data=lat_array.astype(float)),
        "longitude": MaskedArray(data=lon_array.astype(float)),
    }

    return data_arrays


def load_vco_data_arrays(img_file_path: Path, geo_file_path: Path) -> ImgGeoDataArrays:
    img_array = sd_util.load_fits_data(img_file_path, 1)
    ina_array, ema_array, lat_array, lon_array = sd_util.load_fits_data(
        geo_file_path,
        ["Incidence angle", "Emission angle", "Latitude", "Longitude"],
    )

    lon_array = sd_util.fix_360_longitude(lon_array)

    data_arrays: ImgGeoDataArrays = {
        "image": MaskedArray(data=img_array.astype(float)),
        "incidence_angle": MaskedArray(data=ina_array.astype(float)),
        "emission_angle": MaskedArray(data=ema_array.astype(float)),
        "latitude": MaskedArray(data=lat_array.astype(float)),
        "longitude": MaskedArray(data=lon_array.astype(float)),
    }

    return data_arrays


def generate_jno_jnc_patches(
    archive: "JnoJncArchive",
    dataset: "SatelliteDataset",
    patch_scale_km: float,
    patch_resolution: int,
    global_normalization: bool = False,
) -> None:
    output_dir_path = get_patch_dataset_path(
        dataset.name, patch_scale_km, patch_resolution
    )
    output_dir_path.mkdir(parents=True, exist_ok=True)

    patch_generator = JnoJncPatchGenerator(patch_scale_km, patch_resolution)
    patch_generator.generate(dataset, output_dir_path)

    patch_dataset_info_dict = {
        "scale_km": patch_scale_km,
        "resolution": patch_resolution,
        "labels": [],
    }
    patch_dataset_info_json_path = output_dir_path / "info.json"

    with open(patch_dataset_info_json_path, "w") as patch_dataset_info_json:
        json.dump(
            patch_dataset_info_dict,
            patch_dataset_info_json,
            indent=user_config.JSON_INDENT,
        )


def get_patch_dataset_path(
    dataset_name: str, patch_scale_km: float, patch_resolution: int
) -> "Path":
    output_dir_name = f"{dataset_name}_s{patch_scale_km:g}-r{patch_resolution}"
    output_dir_path = config.PATCH_DATASETS_DIR_PATH / dataset_name / output_dir_name

    return output_dir_path
