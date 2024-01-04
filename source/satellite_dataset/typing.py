from typing import TypedDict

from numpy import ndarray


class SatelliteDataArchive(TypedDict):
    name: str
    type: str
    spice: str | None


class DownloadConfig(TypedDict):
    archive: str
    instrument: str
    wavelengths: list[str]


class SatelliteDataset(TypedDict):
    name: str
    path: str
    archive: str


class ImgGeoDataArrays(TypedDict):
    image: ndarray
    latitude: ndarray
    longitude: ndarray
    incidence_angle: ndarray
    emission_angle: ndarray
