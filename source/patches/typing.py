from typing import TypedDict

from numpy import ndarray


class ImgGeoDataArrays(TypedDict):
    image: ndarray
    latitude: ndarray
    longitude: ndarray
    local_time: ndarray
    incidence_angle: ndarray
    emission_angle: ndarray
