from typing import Literal, TypeAlias

ReductionMethod: TypeAlias = Literal["tsne", "pca"] | None
ClusteringMethod: TypeAlias = Literal["kmeans", "hdbscan", "hac"]
