import typing

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader

import source.patch_dataset.config as pd_config
import source.plotting as plotting

if typing.TYPE_CHECKING:
    from source.patch_dataset.dataset import PatchDataset


def plot_dataset(dataset: "PatchDataset", num_patches: int | None = None) -> None:
    dataset_size = len(dataset)
    num_patches = dataset_size if num_patches is None else num_patches

    rand_indices = np.arange(dataset_size)
    np.random.shuffle(rand_indices)

    data_loader = DataLoader(dataset, batch_size=num_patches, sampler=rand_indices)
    patches_tensor: Tensor = next(iter(data_loader))
    patch_images = list(patches_tensor.movedim(1, -1).numpy())

    patch_longitudes = dataset.longitudes[rand_indices]
    patch_latitudes = dataset.latitudes[rand_indices]
    patch_local_times = dataset.local_times[rand_indices]

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

    output_file_name = f"{dataset.name}_{dataset.version_name}_scatter.png"
    output_file_path = pd_config.DATASET_PLOTS_DIR_PATH / output_file_name
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_file_path, bbox_inches="tight")
