from argparse import ArgumentParser, Namespace
from itertools import permutations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from sklearn import metrics
from sklearn.manifold import TSNE

import source.config as config
import source.patch_dataset.dataset as pd_dataset
import source.plotting as plotting
import user.config as user_config
from source.patch_dataset.dataset import PatchDataset

if user_config.ENABLE_TEX_PLOTS:
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.titlepad"] = 8


PLOTS_DIR_NAME = "classification"


def main() -> None:
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name
    version_name: str | None = input_args.version
    num_classes: int | None = input_args.num_classes

    dataset = pd_dataset.get(name=dataset_name, version_name=version_name)
    num_classes = len(dataset.label_names) if dataset.has_labels else num_classes
    class_labels, encoded_dataset = dataset.classify(num_classes=num_classes)

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

    plot_classification_scatter(
        class_labels,
        dataset,
        plots_dir_path / f"{dataset.name}_{dataset.version_name}_class-scatter.png",
    )
    plot_classification_tsne(
        class_labels,
        encoded_dataset,
        dataset,
        plots_dir_path / f"{dataset.name}_{dataset.version_name}_class-tsne.png",
    )
    plot_class_examples(
        class_labels,
        dataset,
        plots_dir_path / f"{dataset.name}_{dataset.version_name}_class-examples.pdf",
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
        "-n",
        "--num_classes",
        type=int,
        help="optional number of classes (ignored if dataset has labels)",
    )

    return arg_parser.parse_args()


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

    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    ax1, ax2 = axes

    for label, color in zip(unique_labels, colors):
        label_mask = class_labels == label
        label_longitudes = patch_longitudes[label_mask]
        label_latitudes = patch_latitudes[label_mask]
        label_local_times = patch_local_times[label_mask]

        ax1.scatter(label_longitudes, label_latitudes, color=color, s=9)
        ax2.scatter(
            label_local_times,
            label_latitudes,
            color=color,
            s=9,
            label=f"Class {label}",
        )

    ax1.set_xlim(-180, 180)
    ax1.set_xlabel("Longitude [deg]")
    ax1.set_xticks(np.arange(-180, 181, 30))

    ax2.legend(fancybox=False)
    ax2.set_xlim(24, 0)  # Might need parameter for planet rotation direction
    ax2.set_xlabel("Local time [h]")
    ax2.set_xticks(np.arange(0, 25, 2))

    for ax in axes:
        ax.grid(linewidth=0.5, alpha=0.1)
        ax.set_ylim(-90, 90)
        ax.set_ylabel("Latitude [deg]")
        ax.tick_params(direction="in", top=True, right=True)

    fig.savefig(file_path, bbox_inches="tight", dpi=user_config.PLOT_DPI)
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
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
        ax1, ax2 = axes

        ax1.set_title("True labels")
        ax2.set_title("Class labels")

        for label, color in zip(unique_labels, colors):
            true_label_points = tsne_map[dataset.labels == label]
            class_label_points = tsne_map[class_labels == label]
            label_name = dataset.label_names[label]

            ax1.scatter(
                true_label_points[:, 0],
                true_label_points[:, 1],
                color=color,
                label=label_name,
                s=9,
            )
            ax2.scatter(
                class_label_points[:, 0], class_label_points[:, 1], color=color, s=9
            )

        ax1.legend(
            loc="upper center",
            bbox_to_anchor=(1.025, -0.025),
            ncol=dataset.num_labels,
            fancybox=False,
            handletextpad=0,
            ncols=3,
            columnspacing=0.5,
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        ax1, ax2 = axes

        ax1.set_title("Images")
        plotting.imscatter(ax1, dataset, tsne_map[:, 0], tsne_map[:, 1], cmap="gray")

        ax2.set_title("Class labels")

        for label, color in zip(unique_labels, colors):
            label_points = tsne_map[class_labels == label]

            ax2.scatter(
                label_points[:, 0],
                label_points[:, 1],
                color=color,
                label=f"Class {label}",
                s=18,
            )

        ax2.legend(fancybox=False, handletextpad=0)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(wspace=0.05)
    fig.savefig(file_path, bbox_inches="tight", dpi=user_config.PLOT_DPI)
    plt.close()


def plot_class_examples(
    class_labels: ndarray, dataset: PatchDataset, file_path: Path, num_examples: int = 8
) -> None:
    print("Plotting class examples...")

    unique_labels = np.unique(class_labels)
    num_labels = unique_labels.size
    fig_width = 9
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

    fig.savefig(file_path, bbox_inches="tight")
    plt.close()


def get_matched_labels(class_labels: ndarray, true_labels: ndarray) -> ndarray:
    class_to_true_maps = np.asarray(list(permutations(np.unique(true_labels))))

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
