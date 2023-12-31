import json
import typing
from collections.abc import Sequence
from pathlib import Path

from astropy.io import fits
from astropy.io.fits.hdu.base import _BaseHDU

import user.config as config
from source.satellite_dataset.typing import SatelliteDataset
from source.utility import user_confirm


def load_satellite_dataset(
    dataset_name: str | None = None,
) -> tuple[str, SatelliteDataset]:
    with open(config.SATELLITE_DATASETS_JSON_PATH, "r") as json_file:
        satellite_datasets: dict[str, SatelliteDataset] = json.load(json_file)

    if dataset_name is None:
        if user_confirm("Display available datasets?"):
            print("\nAvailable datasets:\n-------------------")

            for dataset_name in satellite_datasets:
                print(dataset_name)

            print()

        dataset_name = input("Enter dataset name: ")

    return dataset_name, satellite_datasets[dataset_name]


@typing.overload
def load_fits_hdu_or_hdus(file_path: Path, hdu_key_or_keys: int | str) -> _BaseHDU:
    ...


@typing.overload
def load_fits_hdu_or_hdus(
    file_path: Path, hdu_key_or_keys: Sequence[int | str]
) -> list[_BaseHDU]:
    ...


def load_fits_hdu_or_hdus(
    file_path: Path, hdu_key_or_keys: int | str | Sequence[int | str]
) -> _BaseHDU | list[_BaseHDU]:
    if isinstance(hdu_key_or_keys, Sequence):
        hdus: list[_BaseHDU] = []

        with fits.open(file_path, memmap=False) as file_hdulist:
            for hdu_key in hdu_key_or_keys:
                hdus.append(file_hdulist[hdu_key])  # type: ignore

        return hdus
    else:
        with fits.open(file_path, memmap=False) as file_hdulist:
            return file_hdulist[hdu_key_or_keys]  # type: ignore
