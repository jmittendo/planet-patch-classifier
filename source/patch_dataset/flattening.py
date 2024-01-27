import typing

import numpy as np
from numpy import ndarray
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

import source.utility as util

if typing.TYPE_CHECKING:
    from source.patch_dataset.dataset import PatchDataset


def generate_flat_dataset(dataset: "PatchDataset", blur_sigma: float) -> None:
    try:
        patch_images_dir_path = dataset.get_version_dir_path(version_name="norm-local")
    except FileNotFoundError as error:
        raise ValueError(
            "Flattened dataset can only be generated from a local normalization dataset"
        ) from error

    flat_images_dir_path = patch_images_dir_path.parent / f"flat-s{blur_sigma:g}"
    flat_images_dir_path.mkdir(exist_ok=True)

    for patch_info in tqdm(dataset, desc="Progress"):
        patch_img_file_name = patch_info["file_name"]
        patch_img_file_path = patch_images_dir_path / patch_img_file_name
        patch_img = np.asarray(Image.open(patch_img_file_path)).astype(float)
        flat_patch_img = _flatten_patch(patch_img, blur_sigma)

        Image.fromarray(flat_patch_img).save(flat_images_dir_path / patch_img_file_name)


def _flatten_patch(patch_img: ndarray, blur_sigma: float) -> ndarray:
    flat_img = patch_img - ndimage.gaussian_filter(patch_img, blur_sigma)
    normalized_img = util.get_normalized_img(flat_img)

    return (normalized_img * 255).astype(np.uint8)
