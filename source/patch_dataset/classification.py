import typing
import warnings

from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from source.neural_network.typing import DeviceLike
from source.patch_dataset.typing import ClusteringMethod, ReductionMethod

if typing.TYPE_CHECKING:
    from source.patch_dataset.dataset import PatchDataset


def classify_dataset(
    dataset: "PatchDataset",
    reduction_method: ReductionMethod,
    clustering_method: ClusteringMethod,
    num_classes: int | None,
    pca_dims: int | None = None,
    hdbscan_min_cluster_size: int = 5,
    device: DeviceLike | None = None,
) -> tuple[ndarray, ndarray]:
    encoded_dataset = dataset.encode(device=device)

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
        case _:
            raise ValueError(f"'{clustering_method}' is not a valid clustering method")

    return class_labels, encoded_dataset
