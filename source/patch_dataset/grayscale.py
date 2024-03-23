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
