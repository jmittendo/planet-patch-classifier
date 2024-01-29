import typing

from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

if typing.TYPE_CHECKING:
    from source.patch_dataset.dataset import PatchDataset


def classify_dataset(dataset: "PatchDataset") -> tuple[ndarray, ndarray]:
    encoded_dataset = dataset.encode()

    tsne = TSNE(n_components=3)
    reduced_dataset = tsne.fit_transform(encoded_dataset)

    k_means = KMeans(n_clusters=4)

    return k_means.fit_predict(reduced_dataset), encoded_dataset
