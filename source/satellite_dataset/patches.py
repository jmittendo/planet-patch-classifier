from pathlib import Path

import pandas as pd
from pandas import DataFrame

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

    dataset_table = pd.read_pickle(table_path)

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
            raise NotImplementedError()
        case "vco":
            img_hdu = sd_util.load_fits_hdu_or_hdus(img_file_path, 1)
            lat_hdu, lon_hdu, lt_hdu, ina_hdu, ema_hdu = sd_util.load_fits_hdu_or_hdus(
                geo_file_path,
                [
                    "Latitude",
                    "Longitude",
                    "Local time",
                    "Incidence angle",
                    "Emission angle",
                ],
            )

            data_arrays: ImgGeoDataArrays = {
                "image": img_hdu.data,
                "latitude": lat_hdu.data,
                "longitude": lon_hdu.data,
                "local_time": lt_hdu.data,
                "incidence_angle": ina_hdu.data,
                "emission_angle": ema_hdu,
            }  # type: ignore

            return data_arrays
        case _:
            raise ValueError(
                "Can not load data arrays for unknown dataset archive "
                f"{dataset_archive}"
            )


def _passes_resolution_threshold(
    img_max_resolution: float, patch_resolution: float
) -> bool:
    return img_max_resolution / patch_resolution < ucfg.PATCH_RESOLUTION_TOLERANCE
