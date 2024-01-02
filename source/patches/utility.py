import typing
from collections.abc import Sequence
from pathlib import Path

from astropy.io import fits
from astropy.io.fits.hdu.base import _BaseHDU


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
