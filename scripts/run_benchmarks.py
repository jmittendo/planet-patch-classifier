import warnings
from datetime import datetime
from itertools import permutations

import numpy as np
from numpy import ndarray
from pandas import DataFrame
from sklearn import metrics

import source.config as config
import source.neural_network.config as nn_config
import source.patch_dataset.dataset as pd_dataset
from source.patch_dataset.dataset import PatchDataset

ITERATIONS = 3
DATASET_NAME = "cloud-imvn-1.0"
DATASET_VERSION = "default"
ENCODER_MODELS = ["simple", "autoencoder"]
# ENCODER_MODELS = ["simple", "autoencoder", "simclr"]
REDUCTION_METHODS = ["tsne", "pca", None]
CLUSTERING_METHODS = ["kmeans", "hac"]
PCA_DIM_VALUES = [256, 64, 16]
OUTPUT_FILE_NAME = f"benchmark-results_i{ITERATIONS}_{datetime.now()}.pkl"


def main() -> None:
    dataset = pd_dataset.get(name=DATASET_NAME, version_name=DATASET_VERSION)
    num_classes = len(dataset.label_names)

    table = DataFrame(
        columns=(
            "encoder_model",
            "checkpoint_name",
            "reduction_method",
            "pca_dims",
            "clustering_method",
            "mean_adj_rand_score",
            "std_adj_rand_score",
            "err_adj_rand_score",
            "mean_total_accuracy",
            "std_total_accuracy",
            "err_total_accuracy",
        )
    )

    table_file_path = config.DATA_DIR_PATH / "benchmark-results" / OUTPUT_FILE_NAME
    table_file_path.parent.mkdir(exist_ok=True, parents=True)

    table_row_index = 0

    for encoder_model in ENCODER_MODELS:
        checkpoint_path = (
            None
            if encoder_model == "simple"
            else nn_config.CHECKPOINTS_DIR_PATH
            / encoder_model
            / f"{encoder_model}_{DATASET_NAME}_{DATASET_VERSION}.pt"
        )

        for reduction_method in REDUCTION_METHODS:
            pca_dim_values = PCA_DIM_VALUES if reduction_method == "pca" else [None]

            for pca_dims in pca_dim_values:
                for clustering_method in CLUSTERING_METHODS:
                    adj_rand_scores: list[float] = []
                    total_accuracies: list[float] = []

                    for iteration in range(ITERATIONS):
                        print(
                            f"\n{encoder_model = }, {reduction_method = }, "
                            f"{pca_dims = }, {clustering_method = }, {iteration = }"
                        )

                        class_labels, encoded_dataset = dataset.classify(
                            reduction_method,  # type: ignore
                            clustering_method,  # type: ignore
                            num_classes,
                            pca_dims=pca_dims,
                            encoder_model=encoder_model,
                            checkpoint_path=checkpoint_path,
                        )

                        class_labels = get_matched_labels(class_labels, dataset.labels)

                        adj_rand_score = metrics.adjusted_rand_score(
                            dataset.labels, class_labels
                        )
                        conf_matrix = metrics.confusion_matrix(
                            dataset.labels, class_labels
                        )
                        class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(
                            axis=1
                        )
                        total_accuracy = np.average(
                            class_accuracies, weights=conf_matrix.sum(axis=1)
                        )

                        adj_rand_scores.append(adj_rand_score)
                        total_accuracies.append(total_accuracy)

                    mean_adj_rand_score = np.mean(adj_rand_scores)
                    std_adj_rand_score = np.std(adj_rand_scores)
                    err_adj_rand_score = std_adj_rand_score / np.sqrt(ITERATIONS)

                    mean_total_accuracy = np.mean(total_accuracies)
                    std_total_accuracy = np.std(total_accuracies)
                    err_total_accuracy = std_total_accuracy / np.sqrt(ITERATIONS)

                    table.loc[table_row_index] = [
                        encoder_model,
                        None if checkpoint_path is None else checkpoint_path.name,
                        reduction_method,
                        pca_dims,
                        clustering_method,
                        mean_adj_rand_score,
                        std_adj_rand_score,
                        err_adj_rand_score,
                        mean_total_accuracy,
                        std_total_accuracy,
                        err_total_accuracy,
                    ]

                    table.to_pickle(table_file_path)

                    table_row_index += 1


def get_file_name_base(
    dataset: PatchDataset,
    reduction_method: str | None,
    clustering_method: str,
    num_classes: int | None,
    pca_dims: int | None,
    hdbscan_min_cluster_size: int,
) -> str:
    file_name_base = f"{dataset.name}_{dataset.version_name}_r-{reduction_method}"

    if reduction_method == "pca":
        file_name_base += f"-{pca_dims}"

    file_name_base += f"_c-{clustering_method}"

    if clustering_method == "hdbscan":
        file_name_base += f"-{hdbscan_min_cluster_size}"
    else:
        file_name_base += f"-{num_classes}"

    return file_name_base


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