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

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional
from tqdm import tqdm

import source.patch_dataset.config as pd_config
import source.utility as util

if typing.TYPE_CHECKING:
    from source.patch_dataset import PatchDataset


def generate_flat_version(
    dataset: "PatchDataset", blur_sigma: float, contrast: float
) -> None:
    flat_patches_dir_path = (
        dataset.versions_dir_path / f"flat-s{blur_sigma:g}-c{contrast:g}"
    )
    flat_patches_dir_path.mkdir(exist_ok=True)

    for patch_tensor, patch_file_name in tqdm(
        zip(dataset, dataset.file_names), desc="Progress", total=len(dataset)
    ):
        flat_patch_tensor = _flatten_patch(patch_tensor, blur_sigma)
        standardized_patch_tensor = _standardize_patch(flat_patch_tensor, contrast)
        normalized_patch_tensor = util.get_normalized_img(standardized_patch_tensor)

        normalized_patch_array = (
            (normalized_patch_tensor.movedim(0, -1) * 255).byte().numpy()
        )

        if normalized_patch_array.shape[-1] == 1:
            normalized_patch_array = normalized_patch_array[:, :, 0]

        flat_patch_file_path = flat_patches_dir_path / patch_file_name
        Image.fromarray(normalized_patch_array).save(flat_patch_file_path)


def _flatten_patch(patch_tensor: Tensor, blur_sigma: float) -> Tensor:
    flat_patch_tensor = patch_tensor - _blur_patch(patch_tensor, blur_sigma)

    return flat_patch_tensor


def _blur_patch(patch_tensor: Tensor, blur_sigma: float) -> Tensor:
    kernel_size = int(blur_sigma * pd_config.BLUR_KERNEL_MULTIPLIER * 2 + 1)

    blurred_patch_tensor = functional.gaussian_blur(
        patch_tensor, kernel_size, sigma=blur_sigma  # type: ignore
    )

    return blurred_patch_tensor


def _standardize_patch(patch_tensor: Tensor, contrast: float) -> Tensor:
    patch_mean = patch_tensor.mean()
    patch_std = patch_tensor.std()
    standardized_patch_tensor = (patch_tensor - patch_mean) / patch_std * contrast

    return torch.clip(standardized_patch_tensor, -1, 1)
