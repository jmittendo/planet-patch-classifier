from argparse import ArgumentParser, Namespace

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy import ndarray
from sklearn.manifold import TSNE

import source.config as config
import source.patch_dataset.dataset as pd_dataset
import user.config as user_config
from source.patch_dataset.dataset import PatchDataset

if user_config.ENABLE_TEX_PLOTS:
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.titlepad"] = 8


def main() -> None:
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name
    version_name: str | None = input_args.version

    dataset = pd_dataset.get(name=dataset_name, version_name=version_name)
    class_labels, encoded_dataset = dataset.classify()

    plot_classification_scatter(
        class_labels,
        dataset,
        f"{dataset.name}_{dataset.version_name}_class-scatter.png",
    )
    plot_classification_tsne(
        class_labels,
        encoded_dataset,
        f"{dataset.name}_{dataset.version_name}_class-tsne.png",
    )


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=("Generate classes for a patch dataset."),
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")
    arg_parser.add_argument(
        "version", nargs="?", help="version of the dataset to classify"
    )

    return arg_parser.parse_args()


def plot_classification_scatter(
    class_labels: ndarray, dataset: PatchDataset, file_name: str
) -> None:
    print("Plotting classification scatter...")

    cmap = mpl.colormaps["gist_rainbow"]
    unique_labels, _ = np.unique(class_labels, return_counts=True)
    colors = cmap(np.linspace(0, 1, unique_labels.size))

    patch_longitudes = dataset.longitudes
    patch_latitudes = dataset.latitudes
    patch_local_times = dataset.local_times

    if np.isnan(patch_longitudes).any():
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    axes[0].scatter(patch_longitudes, patch_latitudes, c=colors[class_labels], s=2)
    axes[0].set_xlim(-180, 180)
    axes[0].set_xlabel("Longitude [deg]")
    axes[0].set_xticks(np.arange(-180, 181, 30))

    axes[1].scatter(patch_local_times, patch_latitudes, c=colors[class_labels], s=2)
    axes[1].set_xlim(24, 0)  # Might need parameter for planet rotation direction
    axes[1].set_xlabel("Local time [h]")
    axes[1].set_xticks(np.arange(0, 25, 2))

    for ax in axes:
        ax.grid(linewidth=0.5, alpha=0.1)
        ax.set_ylim(-90, 90)
        ax.set_ylabel("Latitude [deg]")
        ax.tick_params(direction="in", top=True, right=True)

    output_file_path = config.PLOTS_DIR_PATH / file_name
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_file_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_classification_tsne(
    class_labels: ndarray, encoded_dataset: ndarray, file_name: str
) -> None:
    print("Plotting classification TSNE...")

    cmap = mpl.colormaps["gist_rainbow"]
    unique_labels, _ = np.unique(class_labels, return_counts=True)
    colors = cmap(np.linspace(0, 1, unique_labels.size))

    tsne = TSNE()
    tsne_map = tsne.fit_transform(encoded_dataset)

    ax: Axes
    fig, ax = plt.subplots(figsize=(9, 9))

    ax.scatter(tsne_map[:, 0], tsne_map[:, 1], c=colors[class_labels], s=2)
    ax.tick_params(direction="in", top=True, right=True)

    output_file_path = config.PLOTS_DIR_PATH / file_name
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_file_path, bbox_inches="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
