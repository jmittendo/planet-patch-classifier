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

from PIL import Image
from torchvision.transforms import functional
from tqdm import tqdm

if typing.TYPE_CHECKING:
    from source.patch_dataset import PatchDataset


def generate_grayscale_version(dataset: "PatchDataset") -> None:
    grayscale_patches_dir_path = (
        dataset.versions_dir_path / f"{dataset.version_name}_grayscale"
    )
    grayscale_patches_dir_path.mkdir(exist_ok=True)

    for patch_tensor, patch_file_name in tqdm(
        zip(dataset, dataset.file_names), desc="Progress", total=len(dataset)
    ):
        grayscale_patch_tensor = functional.rgb_to_grayscale(patch_tensor)
        grayscale_patch_array = (grayscale_patch_tensor[0] * 255).byte().numpy()

        grayscale_patch_file_path = grayscale_patches_dir_path / patch_file_name
        Image.fromarray(grayscale_patch_array).save(grayscale_patch_file_path)
