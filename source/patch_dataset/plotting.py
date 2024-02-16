import typing

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from torch import Tensor
from torch.utils.data import DataLoader

import source.patch_dataset.config as pd_config
import source.plotting as plotting
import user.config as user_config

if typing.TYPE_CHECKING:
    from source.patch_dataset.dataset import PatchDataset


if user_config.ENABLE_TEX_PLOTS:
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.titlepad"] = 8


def plot_dataset_geometry_scatter(
    dataset: "PatchDataset", num_patches: int | None = None
) -> None:
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

    output_file_name = f"{dataset.name}_{dataset.version_name}_geometry-scatter.png"
    output_file_path = (
        pd_config.DATASET_PLOTS_DIR_PATH
        / dataset.name
        / dataset.version_name
        / output_file_name
    )
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_file_path, bbox_inches="tight", dpi=user_config.PLOT_DPI)


def plot_encoded_dataset_tsne_scatter(dataset: "PatchDataset") -> None:
    encoded_dataset = dataset.encode()

    tsne = TSNE()
    tsne_map = tsne.fit_transform(encoded_dataset)

    if dataset.has_labels:
        if dataset.num_labels > 10:
            cmap = mpl.colormaps["gist_rainbow"]
            colors = cmap(np.linspace(0, 1, dataset.num_labels))
        else:
            cmap = mpl.colormaps["tab10"]
            colors = cmap(np.arange(dataset.num_labels))

        unique_labels = np.unique(dataset.labels)

        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        ax1, ax2 = axes

        ax1.set_title("Images")
        plotting.imscatter(ax1, dataset, tsne_map[:, 0], tsne_map[:, 1], cmap="gray")

        ax2.set_title("True labels")

        for label, color in zip(unique_labels, colors):
            label_points = tsne_map[dataset.labels == label]
            label_name = dataset.label_names[label]

            ax2.scatter(
                label_points[:, 0], label_points[:, 1], color=color, label=label_name
            )

        ax2.legend(fancybox=False, handletextpad=0)

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        fig.subplots_adjust(wspace=0.05)
    else:
        fig, ax = plt.subplots(figsize=(9, 9))

        plotting.imscatter(ax, dataset, tsne_map[:, 0], tsne_map[:, 1], cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])

    output_file_name = f"{dataset.name}_{dataset.version_name}_encoded-tsne-scatter.png"
    output_file_path = (
        pd_config.DATASET_PLOTS_DIR_PATH
        / dataset.name
        / dataset.version_name
        / output_file_name
    )
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_file_path, bbox_inches="tight", dpi=user_config.PLOT_DPI)
