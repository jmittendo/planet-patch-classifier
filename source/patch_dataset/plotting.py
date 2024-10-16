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

import itertools
import typing
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from torch import Tensor
from torch.utils.data import DataLoader

import source.patch_dataset.config as pd_config
import source.satellite_dataset as sd
import user.config as user_config
from source import plotting
from source.satellite_dataset import Planet

if typing.TYPE_CHECKING:
    from source.patch_dataset import PatchDataset


plt.rcParams["axes.linewidth"] = 0.4
plt.rcParams["patch.linewidth"] = 0.4
plt.rcParams["grid.linewidth"] = 0.4
plt.rcParams["xtick.major.size"] = 2.5
plt.rcParams["xtick.minor.size"] = 1.5
plt.rcParams["xtick.major.width"] = 0.4
plt.rcParams["xtick.minor.width"] = 0.3
plt.rcParams["ytick.major.size"] = 2.5
plt.rcParams["ytick.minor.size"] = 1.5
plt.rcParams["ytick.major.width"] = 0.4
plt.rcParams["ytick.minor.width"] = 0.3
plt.rcParams["font.size"] = user_config.PLOT_FONT_SIZE
plt.rcParams["savefig.dpi"] = user_config.PLOT_DPI
plt.rcParams["text.usetex"] = user_config.PLOT_ENABLE_TEX

if user_config.PLOT_ENABLE_TEX:
    plt.rcParams["font.family"] = "serif"
else:
    plt.rcParams["font.family"] = user_config.PLOT_FONT
    plt.rcParams["mathtext.fontset"] = user_config.PLOT_MATH_FONT


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

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75))
    ax1, ax2 = axes

    plotting.imscatter(
        ax1, patch_images, patch_longitudes, patch_latitudes, cmap="gray", zoom=0.33
    )
    ax1.set_xlabel("Longitude [deg]")
    ax1.set_ylabel("Latitude [deg]")

    plotting.imscatter(
        ax2, patch_images, patch_local_times, patch_latitudes, cmap="gray", zoom=0.33
    )

    try:
        satellite_dataset = sd.get_dataset(dataset.satellite_dataset_name)
        planet_rotation = satellite_dataset.archive.planet.rotation

        if planet_rotation == Planet.Rotation.RETROGRADE:
            ax2.invert_xaxis()
    except KeyError:
        warnings.warn(f"Satellite dataset '{dataset.satellite_dataset_name}' not found")

    ax2.set_xlabel("Local time [h]")
    ax2.set_xticks(np.linspace(8, 16, 5))
    ax2.set_yticklabels([])

    for ax in axes:
        ax.grid(linewidth=0.5, alpha=0.1)
        ax.tick_params(direction="in", top=True, right=True)

    output_file_name = f"{dataset.name}_{dataset.version_name}_geometry-scatter.png"
    output_file_path = (
        pd_config.DATASET_PLOTS_DIR_PATH
        / dataset.name
        / dataset.version_name
        / output_file_name
    )
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout(w_pad=1, pad=0.1)
    fig.savefig(output_file_path)


def plot_encoded_dataset_tsne_scatter(
    dataset: "PatchDataset",
    encoder_model: str,
    encoder_base_model: str,
    checkpoint_path: Path | None = None,
) -> None:
    encoded_dataset = dataset.encode(
        encoder_model, encoder_base_model, checkpoint_path=checkpoint_path
    )

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

        fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.625))
        ax1, ax2 = axes

        ax1.set_title("Images")
        plotting.imscatter(
            ax1, dataset, tsne_map[:, 0], tsne_map[:, 1], cmap="gray", zoom=0.5
        )

        ax2.set_title("True labels")

        markers = itertools.cycle(("o", "v", "^", "<", ">", "*", "D", "X"))

        for label, color in zip(unique_labels, colors):
            label_points = tsne_map[dataset.labels == label]
            label_name = dataset.label_names[label]

            ax2.scatter(
                label_points[:, 0],
                label_points[:, 1],
                color=color,
                label=label_name,
                s=6,
                marker=next(markers),
            )

        ax2.legend(fancybox=False, handletextpad=0)

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout(pad=0.25, w_pad=1)
    else:
        fig, ax = plt.subplots(figsize=(3.5, 3.5))

        plotting.imscatter(
            ax, dataset, tsne_map[:, 0], tsne_map[:, 1], cmap="gray", zoom=0.33
        )
        ax.set_xticks([])
        ax.set_yticks([])

    output_file_name = (
        f"{dataset.name}_{dataset.version_name}_e-{encoder_model}-{encoder_base_model}"
        "_encoded-tsne-scatter.png"
    )
    output_file_path = (
        pd_config.DATASET_PLOTS_DIR_PATH
        / dataset.name
        / dataset.version_name
        / output_file_name
    )
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout(pad=0.1)
    fig.savefig(output_file_path)
