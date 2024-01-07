from typing import Literal, TypedDict

from numpy import ndarray
from numpy.ma import MaskedArray


class SatelliteDataArchive(TypedDict):
    name: str
    type: str
    spice: str | None
    planet_radius_km: float


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


class SphericalData(TypedDict):
    img_values: ndarray
    x_values: ndarray
    y_values: ndarray
    z_values: ndarray
    radius_km: float
    solar_longitude: float


class PatchCoordinate(TypedDict):
    phi: float
    theta: float
    longitude: float
    latitude: float
    local_time: float


class ImgGeoPatchProjection(TypedDict):
    img_values: ndarray
    x_values: ndarray
    y_values: ndarray


type ImgGeoPatchInterpolation = (
    Literal["nearest"] | Literal["linear"] | Literal["cubic"]
)
