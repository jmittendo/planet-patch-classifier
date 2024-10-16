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

from typing import Iterable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from numpy import ndarray
from torch import Tensor


def imscatter(
    ax: Axes,
    images: Iterable,
    x_values: ndarray,
    y_values: ndarray,
    zoom: float = 1,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    for image, x, y in zip(images, x_values, y_values):
        if isinstance(image, Tensor) and image.ndim == 3:
            image = image.movedim(0, -1).numpy()

        downscale_factor = 32 / (image.shape[0] + image.shape[1]) * zoom

        offset_image = OffsetImage(
            image, zoom=downscale_factor, cmap=cmap, clim=(vmin, vmax)
        )
        annotation_bbox = AnnotationBbox(offset_image, (x, y), frameon=False)

        ax.add_artist(annotation_bbox)

    ax.update_datalim(np.column_stack([x_values, y_values]))
    ax.autoscale()
