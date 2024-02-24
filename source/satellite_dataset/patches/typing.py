from typing import Literal, TypeAlias, TypedDict

from numpy import ndarray
from numpy.ma import MaskedArray


class ImgGeoPatchProjection(TypedDict):
    img_values: ndarray
    x_values: ndarray
    y_values: ndarray


class PatchCoordinate(TypedDict):
    phi: float
    theta: float
    longitude: float
    latitude: float
    local_time: float


class SphericalData(TypedDict):
    img_values: ndarray
    x_values: ndarray
    y_values: ndarray
    z_values: ndarray
    radius_km: float
    solar_longitude: float


class ImgGeoDataArrays(TypedDict):
    image: MaskedArray
    latitude: MaskedArray
    longitude: MaskedArray
    incidence_angle: MaskedArray
    emission_angle: MaskedArray


ImgGeoPatchInterpolation: TypeAlias = (
    Literal["nearest"] | Literal["linear"] | Literal["cubic"]
)
