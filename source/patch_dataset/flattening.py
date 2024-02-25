import typing

from PIL import Image
from torch import Tensor
from torchvision.transforms import functional
from tqdm import tqdm

import source.patch_dataset.config as pd_config
import source.utility as util

if typing.TYPE_CHECKING:
    from source.patch_dataset.dataset import PatchDataset


def generate_flat_dataset(dataset: "PatchDataset", blur_sigma: float) -> None:
    flat_patches_dir_path = dataset.versions_dir_path / f"flat-s{blur_sigma:g}"
    flat_patches_dir_path.mkdir(exist_ok=True)

    for patch_tensor, patch_file_name in tqdm(
        zip(dataset, dataset.file_names), desc="Progress", total=len(dataset)
    ):
        flat_patch_tensor = _flatten_patch(patch_tensor, blur_sigma)
        flat_patch_array = (flat_patch_tensor.movedim(0, -1) * 255).byte().numpy()

        if flat_patch_array.shape[-1] == 1:
            flat_patch_array = flat_patch_array[:, :, 0]

        flat_patch_file_path = flat_patches_dir_path / patch_file_name
        Image.fromarray(flat_patch_array).save(flat_patch_file_path)


def _flatten_patch(patch_tensor: Tensor, blur_sigma: float) -> Tensor:
    flat_patch_tensor = patch_tensor - _blur_patch(patch_tensor, blur_sigma)

    return util.get_normalized_img(flat_patch_tensor)


def _blur_patch(patch_tensor: Tensor, blur_sigma: float) -> Tensor:
    kernel_size = int(blur_sigma * pd_config.BLUR_KERNEL_MULTIPLIER * 2 + 1)

    blurred_patch_tensor = functional.gaussian_blur(
        patch_tensor, kernel_size, sigma=blur_sigma  # type: ignore
    )

    return blurred_patch_tensor
