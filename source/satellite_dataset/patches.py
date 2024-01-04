from pathlib import Path

import pandas as pd
from pandas import DataFrame
from planetaryimage import PDS3Image

import source.satellite_dataset.config as sdcfg
import source.satellite_dataset.table as sd_table
import source.satellite_dataset.utility as sd_util
import user.config as ucfg
from source.satellite_dataset.typing import ImgGeoDataArrays, SatelliteDataset


def generate_patches(
    dataset: SatelliteDataset,
    dataset_name: str,
    patch_scale_km: float,
    patch_resolution: int,
    regenerate_table: bool,
) -> None:
    table_path = sdcfg.DATASET_TABLES_DIR_PATH / f"{dataset_name}.pkl"

    if regenerate_table or not table_path.is_file():
        sd_table.generate_dataset_table(dataset, table_path)

    dataset_table: DataFrame = pd.read_pickle(table_path)

    # Spatial resolution of a patch in m/px (ignoring projection effects / distortions)
    patch_resolution_mpx = patch_scale_km * 1000 / patch_resolution

    dataset_archive = dataset["archive"]

    match dataset_archive:
        case "vex-vmc" | "vco":
            _generate_img_geo_patches(
                dataset_archive, dataset_table, patch_resolution_mpx
            )
        case _:
            raise ValueError(
                "No patch generation script implemented for dataset archive "
                f"'{dataset_archive}'"
            )


def _generate_img_geo_patches(
    dataset_archive: str, dataset_table: DataFrame, patch_resolution_mpx: float
) -> None:
    for row_index, row_data in dataset_table.iterrows():
        img_max_resolution_mpx: float = row_data["max_resolution_mpx"]

        if not _passes_resolution_threshold(
            img_max_resolution_mpx, patch_resolution_mpx
        ):
            continue

        img_file_path = Path(row_data["img_file_path"])
        geo_file_path = Path(row_data["geo_file_path"])

        img_geo_data_arrays = _load_img_geo_data_arrays(
            dataset_archive, img_file_path, geo_file_path
        )


def _load_img_geo_data_arrays(
    dataset_archive: str, img_file_path: Path, geo_file_path: Path
) -> ImgGeoDataArrays:
    match dataset_archive:
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
                lat_array,  # Latitude data
                lon_array,  # Longitude data
                lt_array,  # Local time data
                ina_array,  # Incidence angle data
                ema_array,  # Emission angle data
            ) = sd_util.load_fits_data(
                geo_file_path,
                [
                    "Latitude",
                    "Longitude",
                    "Local time",
                    "Incidence angle",
                    "Emission angle",
                ],
            )
        case _:
            raise ValueError(
                "Can not load data arrays for unknown dataset archive "
                f"{dataset_archive}"
            )

    data_arrays: ImgGeoDataArrays = {
        "image": img_array,
        "latitude": lat_array,
        "longitude": lon_array,
        "local_time": lt_array,
        "incidence_angle": ina_array,
        "emission_angle": ema_array,
    }

    return data_arrays


def _passes_resolution_threshold(
    img_max_resolution: float, patch_resolution: float
) -> bool:
    return img_max_resolution / patch_resolution < ucfg.PATCH_RESOLUTION_TOLERANCE
