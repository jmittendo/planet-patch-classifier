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
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import spiceypy as spice
from astropy.io import fits
from astropy.io.fits import Header
from numpy import ndarray
from planetaryimage import PDS3Image


@typing.overload
def load_fits_data(file_path: Path, hdu_key_or_keys: int | str) -> ndarray: ...


@typing.overload
def load_fits_data(
    file_path: Path, hdu_key_or_keys: Sequence[int | str]
) -> list[ndarray]: ...


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


def load_pds3_data(file_path: Path) -> ndarray:
    return PDS3Image.open(file_path.as_posix()).data


def load_spice_kernels(kernels_dir_path: Path) -> None:
    for subdir in kernels_dir_path.iterdir():
        if subdir.is_file() or subdir.name == "mk":
            continue

        for file_path in subdir.iterdir():
            if file_path.is_dir() or file_path.suffix == ".txt":
                continue

            spice.furnsh(file_path.resolve().as_posix())


@typing.overload
def fix_360_longitude(longitude: float) -> float: ...


@typing.overload
def fix_360_longitude(longitude: ndarray) -> ndarray: ...


def fix_360_longitude(longitude: float | ndarray) -> float | ndarray:
    # Convert longitudes from range [0°, 360°] to range [-180°, 180°]

    return (longitude + 180) % 360 - 180


def get_zy_rotation_matrix(z_angle: float, y_angle: float) -> ndarray:
    sin_z_angle = np.sin(z_angle)
    cos_z_angle = np.cos(z_angle)
    sin_y_angle = np.sin(y_angle)
    cos_y_angle = np.cos(y_angle)

    z_rot_matrix = np.asarray(
        [
            [cos_z_angle, -sin_z_angle, 0],
            [sin_z_angle, cos_z_angle, 0],
            [0, 0, 1],
        ]
    )

    y_rot_matrix = np.asarray(
        [
            [cos_y_angle, 0, sin_y_angle],
            [0, 1, 0],
            [-sin_y_angle, 0, cos_y_angle],
        ]
    )

    return np.dot(y_rot_matrix, z_rot_matrix)


@typing.overload
def longitude_to_local_time(longitude: float, solar_longitude: float) -> float: ...


@typing.overload
def longitude_to_local_time(longitude: ndarray, solar_longitude: float) -> ndarray: ...


def longitude_to_local_time(
    longitude: float | ndarray, solar_longitude: float
) -> float | ndarray:
    # NOTE: Might need a parameter for planet rotation direction if not Venus

    return (solar_longitude - longitude + 180) % 360 / 360 * 24
