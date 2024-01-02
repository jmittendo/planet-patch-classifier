import json
import typing
from collections.abc import Sequence
from pathlib import Path

from astropy.io import fits
from astropy.io.fits import Header
from numpy import ndarray

import source.satellite_dataset.config as sdcfg
import source.utility as util
from source.satellite_dataset.typing import SatelliteDataset


def load_dataset(dataset_name: str | None = None) -> tuple[str, SatelliteDataset]:
    with open(sdcfg.DATASETS_JSON_PATH, "r") as json_file:
        satellite_datasets: dict[str, SatelliteDataset] = json.load(json_file)

    if dataset_name is None:
        if util.user_confirm("Display available datasets?"):
            print("\nAvailable datasets:\n-------------------")

            for dataset_name in satellite_datasets:
                print(dataset_name)

            print()

        dataset_name = input("Enter dataset name: ")

    return dataset_name, satellite_datasets[dataset_name]


@typing.overload
def load_fits_data(file_path: Path, hdu_key_or_keys: int | str) -> ndarray:
    ...


@typing.overload
def load_fits_data(
    file_path: Path, hdu_key_or_keys: Sequence[int | str]
) -> list[ndarray]:
    ...


def load_fits_data(
    file_path: Path, hdu_key_or_keys: int | str | Sequence[int | str]
) -> ndarray | list[ndarray]:
    if isinstance(hdu_key_or_keys, Sequence):
        data_list: list[ndarray] = []

        with fits.open(file_path, memmap=False) as file_hdulist:
            for hdu_key in hdu_key_or_keys:
                data_list.append(file_hdulist[hdu_key].data)  # type: ignore

        return data_list
    else:
        with fits.open(file_path, memmap=False) as file_hdulist:
            return file_hdulist[hdu_key_or_keys].data  # type: ignore


def load_fits_header(file_path: Path, hdu_key: int | str) -> Header:
    with fits.open(file_path, memmap=False) as file_hdulist:
        return file_hdulist[hdu_key].header  # type: ignore
