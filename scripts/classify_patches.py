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
import warnings
from argparse import ArgumentParser, Namespace
from itertools import permutations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from sklearn import metrics
from sklearn.manifold import TSNE

import source.neural_network as nn
import source.patch_dataset as pd
import source.satellite_dataset as sd
import user.config as user_config
from source import config, plotting
from source.patch_dataset import PatchDataset
from source.satellite_dataset import Planet

PLOTS_DIR_NAME = "classification"


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


def main() -> None:
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name
    version_name: str | None = input_args.version
    reduction_method: str | None = input_args.reduction_method
    clustering_method: str | None = input_args.clustering_method
    encoder_model: str | None = input_args.encoder
    encoder_base_model: str | None = input_args.base_model
    num_classes: int | None = input_args.classes
    pca_dims: int | None = input_args.pca_dims
    hdbscan_min_cluster_size: int | None = input_args.hdbscan_min_cluster_size
    device: str | None = input_args.device

    dataset = pd.get_dataset(name=dataset_name, version_name=version_name)
    num_classes = len(dataset.label_names) if dataset.has_labels else num_classes

    if reduction_method is None:
        reduction_method = input("Enter reduction method ('tsne', 'pca', 'none'): ")

    reduction_method = None if reduction_method == "none" else reduction_method

    if clustering_method is None:
        clustering_method = input(
            "Enter clustering method ('kmeans', 'hac', 'hdbscan'): "
        )

    if clustering_method != "hdbscan" and num_classes is None:
        num_classes = int(input("Enter number of classes: "))

    if reduction_method == "pca" and pca_dims is None:
        pca_dims = int(input("Enter PCA dimensions: "))

    if clustering_method == "hdbscan" and hdbscan_min_cluster_size is None:
        hdbscan_min_cluster_size = int(input("Enter HDBSCAN min cluster size: "))

    hdbscan_min_cluster_size = (
        5 if hdbscan_min_cluster_size is None else hdbscan_min_cluster_size
    )

    if encoder_model is None:
        encoder_model = input(
            "Enter encoder model type ('simple', 'autoencoder', 'simclr'): "
        )

    if encoder_model != "autoencoder" and encoder_base_model is None:
        encoder_base_model = input(
            "Enter encoder base model type "
            "('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'): "
        )

    if encoder_model == "autoencoder" and encoder_base_model != "resnet18":
        warnings.warn("Autoencoder model always uses 'resnet18' base model")
        encoder_base_model = "resnet18"

    checkpoint_path = (
        None
        if encoder_model == "simple"
        else nn.get_checkpoint_path(
            encoder_model, encoder_base_model, dataset.name, dataset.version_name  # type: ignore
        )
    )

    if checkpoint_path is not None and not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"No model checkpoint found for encoder model type '{encoder_model}' and "
            f"dataset '{dataset.name}' with version '{dataset.version_name}'"
        )

    class_labels, encoded_dataset = dataset.classify(
        reduction_method,  # type: ignore
        clustering_method,  # type: ignore
        encoder_model,
        encoder_base_model,  # type: ignore
        num_classes,
        pca_dims=pca_dims,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    if dataset.has_labels:
        class_labels = get_matched_labels(class_labels, dataset.labels)

        adj_rand_score = metrics.adjusted_rand_score(dataset.labels, class_labels)
        conf_matrix = metrics.confusion_matrix(dataset.labels, class_labels)
        class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        total_accuracy = np.average(class_accuracies, weights=conf_matrix.sum(axis=1))

        print(f"\nAdjusted rand score: {adj_rand_score}\n")
        print(f"Confusion matrix:\n{conf_matrix}\n")
        print(f"Class accuracies:\n{class_accuracies}\n")
        print(f"Total accuracy: {total_accuracy}\n")

    plots_dir_path = (
        config.PLOTS_DIR_PATH / PLOTS_DIR_NAME / dataset.name / dataset.version_name
    )
    plots_dir_path.mkdir(parents=True, exist_ok=True)

    file_name_base = get_file_name_base(
        dataset,
        reduction_method,
        clustering_method,
        num_classes,
        pca_dims,
        hdbscan_min_cluster_size,
        encoder_model,
        encoder_base_model,  # type: ignore
    )

    plot_classification_scatter(
        class_labels,
        dataset,
        plots_dir_path / f"{file_name_base}_class-scatter.png",
    )
    plot_classification_tsne(
        class_labels,
        encoded_dataset,
        dataset,
        plots_dir_path / f"{file_name_base}_class-tsne.png",
    )
    plot_class_examples(
        class_labels,
        dataset,
        plots_dir_path / f"{file_name_base}_class-examples.pdf",
    )


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=("Generate classes for a patch dataset."),
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")
    arg_parser.add_argument(
        "version", nargs="?", help="version of the dataset to classify"
    )
    arg_parser.add_argument(
        "reduction_method", nargs="?", help="method to use for dimensionality reduction"
    )
    arg_parser.add_argument(
        "clustering_method", nargs="?", help="method to use for clustering"
    )
    arg_parser.add_argument("encoder", nargs="?", help="model type to use for encoding")
    arg_parser.add_argument("base_model", nargs="?", help="base model type for encoder")
    arg_parser.add_argument(
        "-c",
        "--classes",
        type=int,
        help="optional number of classes (ignored if dataset has labels)",
    )
    arg_parser.add_argument(
        "-p",
        "--pca_dims",
        type=int,
        help=(
            "number of dimensions for dimensionality reduction with PCA "
            "(ignored otherwise)"
        ),
    )
    arg_parser.add_argument(
        "-m",
        "--hdbscan_min_cluster_size",
        type=int,
        help=("min cluster size for clustering with HDBSCAN (ignored otherwise)"),
    )
    arg_parser.add_argument("-d", "--device", help="device to use for encoder model")

    return arg_parser.parse_args()


def get_file_name_base(
    dataset: PatchDataset,
    reduction_method: str | None,
    clustering_method: str,
    num_classes: int | None,
    pca_dims: int | None,
    hdbscan_min_cluster_size: int,
    encoder_model: str,
    encoder_base_model: str,
) -> str:
    file_name_base = f"{dataset.name}_{dataset.version_name}_r-{reduction_method}"

    if reduction_method == "pca":
        file_name_base += f"-{pca_dims}"

    file_name_base += f"_c-{clustering_method}"

    if clustering_method == "hdbscan":
        file_name_base += f"-{hdbscan_min_cluster_size}"
    else:
        file_name_base += f"-{num_classes}"

    file_name_base += f"_e-{encoder_model}-{encoder_base_model}"

    return file_name_base


def plot_classification_scatter(
    class_labels: ndarray, dataset: PatchDataset, file_path: Path
) -> None:
    print("Plotting classification scatter...")

    unique_labels = np.unique(class_labels)
    num_labels = unique_labels.size

    if num_labels > 10:
        cmap = mpl.colormaps["gist_rainbow"]
        colors = cmap(np.linspace(0, 1, num_labels))
    else:
        cmap = mpl.colormaps["tab10"]
        colors = cmap(np.arange(num_labels))

    patch_longitudes = dataset.longitudes
    patch_latitudes = dataset.latitudes
    patch_local_times = dataset.local_times

    if np.isnan(patch_longitudes).any():
        return

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.1))
    ax1, ax2 = axes

    markers = itertools.cycle(("o", "v", "^", "<", ">", "*", "D", "X"))

    for label, color in zip(unique_labels, colors):
        label_mask = class_labels == label
        label_longitudes = patch_longitudes[label_mask]
        label_latitudes = patch_latitudes[label_mask]
        label_local_times = patch_local_times[label_mask]
        marker = next(markers)

        ax1.scatter(
            label_longitudes,
            label_latitudes,
            color=color,
            s=2,
            label=f"Class {label}",
            marker=marker,
        )
        ax2.scatter(label_local_times, label_latitudes, color=color, s=2, marker=marker)

    ax1.set_xlabel("Longitude [deg]")
    ax1.set_ylabel("Latitude [deg]")

    legend = ax1.legend(
        loc="upper center",
        bbox_to_anchor=(1.035, 1.2),
        fancybox=False,
        handletextpad=0,
        ncols=4,
        columnspacing=0.5,
    )
    legend.set_in_layout(False)

    try:
        satellite_dataset = sd.get_dataset(dataset.satellite_dataset_name)
        planet_rotation = satellite_dataset.archive.planet.rotation

        if planet_rotation is Planet.Rotation.RETROGRADE:
            ax2.invert_xaxis()
    except KeyError:
        warnings.warn(f"Satellite dataset '{dataset.satellite_dataset_name}' not found")

    ax2.set_xlabel("Local time [h]")
    ax2.set_xticks(np.linspace(8, 16, 5))
    ax2.set_yticklabels([])

    for ax in axes:
        ax.grid(linewidth=0.5, alpha=0.1)
        ax.tick_params(direction="in", top=True, right=True)

    fig.tight_layout(rect=(0, 0, 1, 0.875), pad=0.1, w_pad=1)
    fig.savefig(file_path)
    plt.close()


def plot_classification_tsne(
    class_labels: ndarray,
    encoded_dataset: ndarray,
    dataset: PatchDataset,
    file_path: Path,
) -> None:
    print("Plotting classification TSNE...")

    unique_labels = np.unique(class_labels)
    num_labels = unique_labels.size

    if num_labels > 10:
        cmap = mpl.colormaps["gist_rainbow"]
        colors = cmap(np.linspace(0, 1, num_labels))
    else:
        cmap = mpl.colormaps["tab10"]
        colors = cmap(np.arange(num_labels))

    tsne = TSNE()
    tsne_map = tsne.fit_transform(encoded_dataset)

    if dataset.has_labels:
        fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.3))
        ax1, ax2 = axes

        ax1.set_title("True labels")
        ax2.set_title("Class labels")

        markers = itertools.cycle(("o", "v", "^", "<", ">", "*", "D", "X"))

        for label, color in zip(unique_labels, colors):
            true_label_points = tsne_map[dataset.labels == label]
            class_label_points = tsne_map[class_labels == label]
            label_name = dataset.label_names[label]
            marker = next(markers)

            ax1.scatter(
                true_label_points[:, 0],
                true_label_points[:, 1],
                color=color,
                label=label_name,
                s=3,
                marker=marker,
            )
            ax2.scatter(
                class_label_points[:, 0],
                class_label_points[:, 1],
                color=color,
                s=3,
                marker=marker,
            )

        legend = ax1.legend(
            loc="upper center",
            bbox_to_anchor=(1.025, -0.025),
            fancybox=False,
            handletextpad=0,
            ncols=3,
            columnspacing=0.5,
        )
        legend.set_in_layout(False)

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout(rect=(0, 0.2, 1, 1), pad=0.1, w_pad=1)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.2))
        ax1, ax2 = axes

        ax1.set_title("Images")
        plotting.imscatter(
            ax1, dataset, tsne_map[:, 0], tsne_map[:, 1], cmap="gray", zoom=0.33
        )

        ax2.set_title("Class labels")

        markers = itertools.cycle(("o", "v", "^", "<", ">", "*", "D", "X"))

        for label, color in zip(unique_labels, colors):
            label_points = tsne_map[class_labels == label]

            ax2.scatter(
                label_points[:, 0],
                label_points[:, 1],
                color=color,
                label=f"Class {label}",
                s=2,
                marker=next(markers),
            )

        legend = ax2.legend(
            loc="upper center",
            bbox_to_anchor=(-0.035, -0.025),
            fancybox=False,
            handletextpad=0,
            ncols=unique_labels.size,
            columnspacing=0.5,
        )
        legend.set_in_layout(False)

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout(rect=(0, 0.15, 1, 1), pad=0.1, w_pad=1)

    fig.savefig(file_path)
    plt.close()


def plot_class_examples(
    class_labels: ndarray, dataset: PatchDataset, file_path: Path, num_examples: int = 8
) -> None:
    print("Plotting class examples...")

    unique_labels = np.unique(class_labels)
    num_labels = unique_labels.size
    fig_width = 3.5
    fig_height = num_labels / num_examples * fig_width

    fig, axes = plt.subplots(num_labels, num_examples, figsize=(fig_width, fig_height))

    for row, label in enumerate(unique_labels):
        axes[row, 0].set_ylabel(f"Class {label}")

        label_img_indices = (class_labels == label).nonzero()[0]

        if label_img_indices.size >= num_examples:
            rand_img_indices = np.random.choice(
                label_img_indices, size=num_examples, replace=False
            )
        else:
            rand_img_indices = label_img_indices

        for column, img_index in enumerate(rand_img_indices):
            img_array = dataset[img_index].movedim(0, -1).numpy()
            axes[row, column].imshow(img_array, cmap="gray")

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(pad=0.1, h_pad=0.25, w_pad=0.5)
    fig.savefig(file_path)
    plt.close()


def get_matched_labels(class_labels: ndarray, true_labels: ndarray) -> ndarray:
    unique_class_labels = np.unique(class_labels)
    unique_true_labels = np.unique(true_labels)

    if unique_class_labels.size != unique_true_labels.size:
        warnings.warn(
            "Number of class labels not equal to number of true labels, "
            "labels will not be matched"
        )
        return class_labels

    class_to_true_maps = np.asarray(list(permutations(unique_true_labels)))

    best_accuracy = 0
    best_map: ndarray | None = None

    for class_to_true_map in class_to_true_maps:
        matched_labels = class_to_true_map[class_labels]
        confusion_matrix = metrics.confusion_matrix(true_labels, matched_labels)
        class_accuracies = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
        total_accuracy = np.average(
            class_accuracies, weights=confusion_matrix.sum(axis=1)
        )

        if total_accuracy > best_accuracy:
            best_accuracy = total_accuracy
            best_map = class_to_true_map

    if best_map is None:
        matched_labels = class_labels
    else:
        matched_labels = best_map[class_labels]

    return matched_labels


if __name__ == "__main__":
    main()
