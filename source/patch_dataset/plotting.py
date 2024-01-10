import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from PIL import Image

import source.patch_dataset.config as pd_config
import source.plotting as plotting
from source.patch_dataset.typing import PatchNormalization

if typing.TYPE_CHECKING:
    from source.patch_dataset.dataset import PatchDataset


def plot_dataset(
    dataset: "PatchDataset",
    patch_normalization: PatchNormalization,
    num_patches: int | None = None,
) -> None:
    patch_info_table_path = dataset.path / "patch-info.pkl"
    patch_info_table: DataFrame = pd.read_pickle(patch_info_table_path)

    num_patches = len(patch_info_table) if num_patches is None else num_patches
    random_rows = patch_info_table.sample(n=num_patches)

    patch_longitudes = random_rows["longitude"].to_numpy()
    patch_latitudes = random_rows["latitude"].to_numpy()
    patch_local_times = random_rows["local_time"].to_numpy()

    patch_images_dir_path = dataset.path / f"{patch_normalization}-normalization"

    if not patch_images_dir_path.is_dir():
        raise FileNotFoundError(
            f"Could not find patch dataset '{patch_images_dir_path.as_posix()}'"
        )

    patch_images: list[ndarray] = []

    for patch_file_name in random_rows["file_name"]:
        patch_img_file_path = patch_images_dir_path / patch_file_name
        patch_image = np.asarray(Image.open(patch_img_file_path))

        patch_images.append(patch_image)

    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    ax1, ax2 = axes

    plotting.imscatter(
        ax1, patch_images, patch_longitudes, patch_latitudes, cmap="gray"
    )
    ax1.set_xlim(-180, 180)
    ax1.set_xlabel("Longitude [deg]")
    ax1.set_xticks(np.arange(-180, 181, 30))

    plotting.imscatter(
        ax2, patch_images, patch_local_times, patch_latitudes, cmap="gray"
    )
    ax2.set_xlim(24, 0)  # Might need parameter for planet rotation direction
    ax2.set_xlabel("Local time [h]")
    ax2.set_xticks(np.arange(0, 25, 2))

    for ax in axes:
        ax.grid(linewidth=0.5, alpha=0.1)
        ax.set_ylim(-90, 90)
        ax.set_ylabel("Latitude [deg]")
        ax.tick_params(direction="in", top=True, right=True)

    output_file_path = pd_config.DATASET_PLOTS_DIR_PATH / f"{dataset.name}_scatter.png"
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_file_path, bbox_inches="tight")
