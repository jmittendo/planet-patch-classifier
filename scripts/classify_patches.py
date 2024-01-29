from argparse import ArgumentParser, Namespace

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

import source.config as config
import source.patch_dataset.dataset as pd_dataset
from source.patch_dataset.dataset import PatchDataset


def main() -> None:
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name
    version_name: str | None = input_args.version

    dataset = pd_dataset.get(name=dataset_name, version_name=version_name)
    class_labels, _ = dataset.classify()

    plot_file_name = f"{dataset.name}_{dataset.version_name}_classes.png"

    plot_classification_scatter(class_labels, dataset, plot_file_name)


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=("Generate a scatter plot for the specified dataset."),
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")
    arg_parser.add_argument(
        "version", nargs="?", help="version of the dataset to use for plotting"
    )

    return arg_parser.parse_args()


def plot_classification_scatter(
    class_labels: ndarray, dataset: PatchDataset, file_name: str
) -> None:
    cmap = mpl.colormaps["gist_rainbow"]
    unique_labels, _ = np.unique(class_labels, return_counts=True)
    colors = cmap(np.linspace(0, 1, unique_labels.size))

    patch_longitudes = dataset.longitudes
    patch_latitudes = dataset.latitudes
    patch_local_times = dataset.local_times

    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    axes[0].scatter(patch_longitudes, patch_latitudes, c=colors[class_labels], s=1)
    axes[0].set_xlim(-180, 180)
    axes[0].set_xlabel("Longitude [deg]")
    axes[0].set_xticks(np.arange(-180, 181, 30))

    axes[1].scatter(patch_local_times, patch_latitudes, c=colors[class_labels], s=1)
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


if __name__ == "__main__":
    main()
