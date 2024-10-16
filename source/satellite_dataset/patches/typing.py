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
