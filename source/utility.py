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

from numpy import ndarray
from torch import Tensor


@typing.overload
def get_normalized_img(img: ndarray) -> ndarray: ...


@typing.overload
def get_normalized_img(img: Tensor) -> Tensor: ...


def get_normalized_img(img: ndarray | Tensor) -> ndarray | Tensor:
    img_min = img.min()
    img_max = img.max()

    return (img - img_min) / (img_max - img_min)
