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
        if isinstance(image, Tensor):
            image = image.movedim(1, -1).numpy()

        if image.ndim == 3:
            if image.shape[0] == 1 and image.shape[1] > 1:
                image = image[0]
            elif image.shape[0] == 3 and image.shape[1] > 3:
                image = np.moveaxis(image, 0, -1)
            elif image.shape[2] == 1 and image.shape[1] > 1:
                image = image[:, :, 0]

        downscale_factor = 32 / (image.shape[0] + image.shape[1]) * zoom

        offset_image = OffsetImage(
            image, zoom=downscale_factor, cmap=cmap, clim=(vmin, vmax)
        )
        annotation_bbox = AnnotationBbox(offset_image, (x, y), frameon=False)

        ax.add_artist(annotation_bbox)

    ax.update_datalim(np.column_stack([x_values, y_values]))
    ax.autoscale()
