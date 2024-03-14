import json
from pathlib import Path
from typing import Iterator, TypedDict

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io

import source.patch_dataset.classification as pd_classification
import source.patch_dataset.config as pd_config
import source.patch_dataset.encoding as pd_encoding
import source.patch_dataset.plotting as pd_plotting
from source.neural_network.typing import DeviceLike
from source.patch_dataset.typing import ClusteringMethod, ReductionMethod


class _PatchDatasetInfoDict(TypedDict):
    scale_km: float | None
    resolution: int
    labels: list[str]


class PatchDataset(Dataset):
    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.satellite_dataset_name = path.parent.name

        self.versions_dir_path = path / "versions"
        self._version_dir_paths_dict = {
            p.name: p for p in self.versions_dir_path.iterdir() if p.is_dir()
        }
        self._version_dir_path: Path | None = None

        dataset_info_json_path = path / "info.json"

        with open(dataset_info_json_path) as dataset_info_json:
            dataset_info_dict: _PatchDatasetInfoDict = json.load(dataset_info_json)

        self.scale_km = dataset_info_dict["scale_km"]
        self.resolution = dataset_info_dict["resolution"]
        self.label_names = dataset_info_dict["labels"]

        table_path = path / "table.pkl"
        self._table: DataFrame = pd.read_pickle(table_path)

        self._images_tensor: Tensor | None = None
        self._mean: Tensor | None = None
        self._std: Tensor | None = None

    @property
    def longitudes(self) -> ndarray:
        return self._get_table_column("longitude")

    @property
    def latitudes(self) -> ndarray:
        return self._get_table_column("latitude")

    @property
    def local_times(self) -> ndarray:
        return self._get_table_column("local_time")

    @property
    def file_names(self) -> list[str]:
        return self._get_table_column("file_name").tolist()

    @property
    def labels(self) -> ndarray:
        return self._get_table_column("label").tolist()

    @property
    def has_labels(self) -> bool:
        return np.isfinite(self.labels).any()  # type: ignore

    @property
    def num_labels(self) -> int:
        if self.has_labels:
            return np.unique(self.labels).size
        else:
            return 0

    @property
    def num_channels(self) -> int:
        return self[0].shape[0]

    @property
    def version_name(self) -> str:
        if self._version_dir_path is None:
            raise ValueError(f"No version set for patch dataset '{self.name}'")

        return self._version_dir_path.name

    @property
    def mean(self) -> Tensor:
        if self._mean is None:
            if self._images_tensor is None:
                sum_tensor = torch.zeros(self[0].shape[0])

                for img_tensor in self:
                    sum_tensor += img_tensor.mean(dim=(1, 2))

                self._mean = sum_tensor / len(self)
            else:
                self._mean = self._images_tensor.mean(dim=(0, 2, 3))

        return self._mean

    @property
    def std(self) -> Tensor:
        if self._std is None:
            if self._images_tensor is None:
                sum_tensor = torch.zeros(self[0].shape[0])
                mean_tensor = self.mean[:, None, None]

                for img_tensor in self:
                    sum_tensor += ((img_tensor - mean_tensor) ** 2).mean(dim=(1, 2))

                self._std = torch.sqrt(sum_tensor / len(self))
            else:
                self._std = self._images_tensor.std(dim=(0, 2, 3))

        return self._std

    def __getitem__(self, index: int) -> Tensor:
        if self._images_tensor is None:
            file_name: str = self._table.iloc[index]["file_name"]

            if self._version_dir_path is None:
                raise ValueError(f"No version set for patch dataset '{self.name}'")

            img_file_path = self._version_dir_path / file_name

            return io.read_image(img_file_path.as_posix()) / 255
        else:
            return self._images_tensor[index]

    def __iter__(self) -> Iterator[Tensor]:
        for index in range(len(self)):
            yield self[index]

    def __len__(self) -> int:
        return len(self._table)

    def set_version(self, version_name: str | None = None) -> None:
        if version_name is None:
            print("\nAvailable versions:\n-------------------")

            for version in self._version_dir_paths_dict:
                print(version)

            print()

            version_name = input("Enter version name: ")

        self._version_dir_path = self._version_dir_paths_dict[version_name]
        self._images_tensor = None

    def load_images(self) -> Tensor:
        if self._images_tensor is None:
            self._images_tensor = torch.stack([img_tensor for img_tensor in self])

        return self._images_tensor

    def plot_geometry_scatter(self, num_patches: int | None = None) -> None:
        pd_plotting.plot_dataset_geometry_scatter(self, num_patches=num_patches)

    def plot_encoded_tsne_scatter(
        self,
        encoder_model: str,
        encoder_base_model: str,
        checkpoint_path: Path | None = None,
    ):
        pd_plotting.plot_encoded_dataset_tsne_scatter(
            self, encoder_model, encoder_base_model, checkpoint_path=checkpoint_path
        )

    def encode(
        self,
        model: str,
        base_model: str,
        checkpoint_path: Path | None = None,
        device: DeviceLike | None = None,
    ) -> ndarray:
        return pd_encoding.encode_dataset(
            self, model, base_model, checkpoint_path=checkpoint_path, device=device
        )

    def classify(
        self,
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
        return pd_classification.classify_dataset(
            self,
            reduction_method,
            clustering_method,
            encoder_model,
            encoder_base_model,
            num_classes,
            pca_dims=pca_dims,
            hdbscan_min_cluster_size=hdbscan_min_cluster_size,
            checkpoint_path=checkpoint_path,
            device=device,
        )

    def _get_table_column(self, column_name: str) -> ndarray:
        if column_name in self._table:
            return self._table[column_name].to_numpy()
        else:
            return np.full(len(self), np.nan)


def _build_dataset_registry() -> dict[str, PatchDataset]:
    print("Loading patch datasets...")

    dataset_registry: dict[str, PatchDataset] = {}

    for satellite_dataset_dir_path in pd_config.DATASETS_DIR_PATH.iterdir():
        if not satellite_dataset_dir_path.is_dir():
            continue

        for patch_dataset_path in satellite_dataset_dir_path.iterdir():
            if not patch_dataset_path.is_dir():
                continue

            dataset_name = patch_dataset_path.name
            dataset = PatchDataset(dataset_name, patch_dataset_path)
            dataset_registry[dataset_name] = dataset

    return dataset_registry


_dataset_registry = _build_dataset_registry()


def get(name: str | None = None, version_name: str | None = None) -> PatchDataset:
    if name is None:
        print("\nAvailable datasets:\n-------------------")

        for dataset_name in _dataset_registry:
            print(dataset_name)

        print()

        name = input("Enter dataset name: ")

    dataset = _dataset_registry[name]
    dataset.set_version(version_name=version_name)

    return dataset
