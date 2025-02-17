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

import typing
import warnings
from pathlib import Path

from numpy import ndarray
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from source.neural_network.typing import DeviceLike
from source.patch_dataset.typing import ClusteringMethod, ReductionMethod

if typing.TYPE_CHECKING:
    from source.patch_dataset import PatchDataset


def classify_dataset(
    dataset: "PatchDataset",
    reduction_method: ReductionMethod,
    clustering_method: ClusteringMethod,
    encoder_model: str,
    encoder_base_model: str,
    num_classes: int | None,
    pca_dims: int | None = None,
    hdbscan_min_cluster_size: int = 5,
    checkpoint_path: Path | None = None,
    device: DeviceLike | None = None,
) -> tuple[ndarray, ndarray]:
    encoded_dataset = dataset.encode(
        encoder_model,
        encoder_base_model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    print("Classifying dataset...")

    if pca_dims is not None and reduction_method != "pca":
        warnings.warn(
            "Atrribute 'pca_dims' is only used with reduction method 'pca' but "
            f"{reduction_method = }"
        )

    match reduction_method:
        case "tsne":
            tsne = TSNE(n_components=3)
            reduced_dataset = tsne.fit_transform(encoded_dataset)
        case "pca":
            pca = PCA(n_components=pca_dims)
            reduced_dataset = pca.fit_transform(encoded_dataset)
        case None:
            reduced_dataset = encoded_dataset
        case _:
            raise ValueError(f"'{reduction_method}' is not a valid reduction method")

    match clustering_method:
        case "kmeans":
            if num_classes is None:
                k_means = KMeans()
            else:
                k_means = KMeans(n_clusters=num_classes)

            class_labels = k_means.fit_predict(reduced_dataset)
        case "hdbscan":
            hdbscan = HDBSCAN(min_samples=hdbscan_min_cluster_size)
            class_labels = hdbscan.fit_predict(reduced_dataset)
        case "hac":
            if num_classes is None:
                hac = AgglomerativeClustering()
            else:
                hac = AgglomerativeClustering(n_clusters=num_classes)

            class_labels = hac.fit_predict(reduced_dataset)
        case _:
            raise ValueError(f"'{clustering_method}' is not a valid clustering method")

    return class_labels, encoded_dataset
