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

import warnings
from datetime import datetime
from itertools import permutations

import numpy as np
from numpy import ndarray
from pandas import DataFrame
from sklearn import metrics

import source.config as config
import source.neural_network as nn
import source.patch_dataset as pd
import user.config as user_config
from source.patch_dataset import PatchDataset


def main() -> None:
    dataset = pd.get_dataset(
        name=user_config.BENCHMARK_DATASET_NAME,
        version_name=user_config.BENCHMARK_DATASET_VERSION,
    )
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

    table_file_name = (
        f"benchmark-results_i{user_config.BENCHMARK_ITERATIONS}_{datetime.now()}.pkl"
    )
    table_file_path = config.DATA_DIR_PATH / "benchmark-results" / table_file_name
    table_file_path.parent.mkdir(exist_ok=True, parents=True)

    table_row_index = 0

    for encoder_model in user_config.BENCHMARK_ENCODER_MODELS:
        checkpoint_path = (
            None
            if encoder_model == "simple"
            else nn.get_checkpoint_path(
                encoder_model,
                user_config.BENCHMARK_ENCODER_BASE_MODEL,
                user_config.BENCHMARK_DATASET_NAME,
                user_config.BENCHMARK_DATASET_VERSION,
            )
        )

        for reduction_method in user_config.BENCHMARK_REDUCTION_METHODS:
            pca_dim_values = (
                user_config.BENCHMARK_PCA_DIM_VALUES
                if reduction_method == "pca"
                else [None]
            )

            for pca_dims in pca_dim_values:
                for clustering_method in user_config.BENCHMARK_CLUSTERING_METHODS:
                    adj_rand_scores: list[float] = []
                    total_accuracies: list[float] = []

                    for iteration in range(user_config.BENCHMARK_ITERATIONS):
                        print(
                            f"\n{encoder_model = }, {reduction_method = }, "
                            f"{pca_dims = }, {clustering_method = }, {iteration = }"
                        )

                        class_labels, encoded_dataset = dataset.classify(
                            reduction_method,  # type: ignore
                            clustering_method,  # type: ignore
                            encoder_model,
                            user_config.BENCHMARK_ENCODER_BASE_MODEL,
                            num_classes,
                            pca_dims=pca_dims,
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
                    err_adj_rand_score = std_adj_rand_score / np.sqrt(
                        user_config.BENCHMARK_ITERATIONS
                    )

                    mean_total_accuracy = np.mean(total_accuracies)
                    std_total_accuracy = np.std(total_accuracies)
                    err_total_accuracy = std_total_accuracy / np.sqrt(
                        user_config.BENCHMARK_ITERATIONS
                    )

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
