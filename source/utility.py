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
