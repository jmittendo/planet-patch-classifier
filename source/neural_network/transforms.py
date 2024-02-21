import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torchvision.transforms import InterpolationMode, RandomRotation
from torchvision.transforms.functional import resized_crop, rotate


class GrayscaleToRGB(Module):
    def forward(self, input_tensor: Tensor):
        return torch.cat([input_tensor] * 3, dim=-3)


class RandomCroppedRotation(RandomRotation):
    def __init__(
        self,
        degrees,
        interpolation=InterpolationMode.NEAREST,
        antialias: str | bool = "warn",
    ) -> None:
        self.antialias = antialias

        super().__init__(degrees, interpolation=interpolation)

    def forward(self, img: Tensor) -> Tensor:
        angle = self.get_params(self.degrees)

        rotated_img = rotate(
            img, angle, interpolation=self.interpolation, fill=0  # type: ignore
        )

        img_size = rotated_img.shape[-1]

        crop_offset, crop_size = self._get_crop_params(img_size, angle)

        cropped_img = resized_crop(
            rotated_img,
            crop_offset,
            crop_offset,
            crop_size,
            crop_size,
            img_size,  # type: ignore
            antialias=self.antialias,
        )

        return cropped_img

    def _get_crop_params(self, img_size: int, angle: float) -> tuple[int, int]:
        x = angle % 90 / 180 * np.pi
        scaling_factor = np.sqrt(2) * np.sin(x + np.pi / 4)

        crop_size = int(img_size // scaling_factor)
        crop_offset = (img_size - crop_size) // 2

        return crop_offset, crop_size
