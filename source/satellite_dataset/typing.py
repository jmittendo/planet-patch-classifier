from typing import TypedDict

from numpy.ma import MaskedArray


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
    image: MaskedArray
    latitude: MaskedArray
    longitude: MaskedArray
    incidence_angle: MaskedArray
    emission_angle: MaskedArray
