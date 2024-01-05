from pathlib import Path

import numpy as np
import pandas as pd
from numpy.ma import MaskedArray
from pandas import DataFrame

import source.satellite_dataset.config as sdcfg
import source.satellite_dataset.table as sd_table
import source.satellite_dataset.utility as sd_util
import user.config as ucfg
from source.satellite_dataset.typing import (
    ImgGeoDataArrays,
    SatelliteDataArchive,
    SatelliteDataset,
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

        img_geo_data_arrays = _load_img_geo_data_arrays(
            archive, img_file_path, geo_file_path
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

            # The invalidity comparison values could be done more "exact" but this
            # should work
            invalid_mask = img_array < 0

            for geo_array in [ina_array, ema_array, lat_array, lon_array]:
                invalid_mask |= geo_array < -1e10

            masked_img_array = MaskedArray(data=img_array, mask=invalid_mask)
            masked_ina_array = MaskedArray(data=ina_array, mask=invalid_mask)
            masked_ema_array = MaskedArray(data=ema_array, mask=invalid_mask)
            masked_lat_array = MaskedArray(data=lat_array, mask=invalid_mask)
            masked_lon_array = MaskedArray(data=lon_array, mask=invalid_mask)
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

            # The invalidity comparison values could be done more "exact" but this
            # should work
            invalid_mask = img_array < -1e38

            for geo_array in [ina_array, ema_array, lat_array, lon_array]:
                invalid_mask |= ~np.isfinite(geo_array)

            masked_img_array = MaskedArray(data=img_array, mask=invalid_mask)
            masked_ina_array = MaskedArray(data=ina_array, mask=invalid_mask)
            masked_ema_array = MaskedArray(data=ema_array, mask=invalid_mask)
            masked_lat_array = MaskedArray(data=lat_array, mask=invalid_mask)
            masked_lon_array = MaskedArray(data=lon_array, mask=invalid_mask)
        case _:
            raise ValueError(
                "Can not load data arrays for unknown dataset archive "
                f"{archive_name}"
            )

    data_arrays: ImgGeoDataArrays = {
        "image": masked_img_array,
        "latitude": masked_lat_array,
        "longitude": masked_lon_array,
        "incidence_angle": masked_ina_array,
        "emission_angle": masked_ema_array,
    }

    return data_arrays


def _passes_resolution_threshold(
    img_max_resolution: float, patch_resolution: float
) -> bool:
    return img_max_resolution / patch_resolution < ucfg.PATCH_RESOLUTION_TOLERANCE
