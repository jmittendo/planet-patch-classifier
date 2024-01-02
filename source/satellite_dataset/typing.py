from typing import TypedDict

from numpy import ndarray


class DownloadConfig(TypedDict):
    archive: str
    instrument: str
    wavelengths: list[str]


class SatelliteDataset(TypedDict):
    path: str
    archive: str


class ImgGeoDataArrays(TypedDict):
    image: ndarray
    latitude: ndarray
    longitude: ndarray
    local_time: ndarray
    incidence_angle: ndarray
    emission_angle: ndarray
