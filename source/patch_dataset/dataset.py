import json
from pathlib import Path
from typing import Iterator, TypedDict

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
import source.utility as util


class _PatchDatasetInfoDict(TypedDict):
    scale_km: float
    resolution: int


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

        table_path = path / "table.pkl"
        self._table: DataFrame = pd.read_pickle(table_path)

        self._images_tensor: Tensor | None = None

    @property
    def longitudes(self) -> ndarray:
        return self._table["longitude"].to_numpy()

    @property
    def latitudes(self) -> ndarray:
        return self._table["latitude"].to_numpy()

    @property
    def local_times(self) -> ndarray:
        return self._table["local_time"].to_numpy()

    @property
    def file_names(self) -> list[str]:
        return self._table["file_name"].to_list()

    @property
    def version_name(self) -> str:
        if self._version_dir_path is None:
            raise ValueError(f"No version set for patch dataset '{self.name}'")

        return self._version_dir_path.name

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
            if util.user_confirm("Display available versions?"):
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

    def plot(
        self,
        num_patches: int | None = None,
    ) -> None:
        pd_plotting.plot_dataset(self, num_patches=num_patches)

    def encode(self) -> ndarray:
        return pd_encoding.encode_dataset(self)

    def classify(self) -> tuple[ndarray, ndarray]:
        return pd_classification.classify_dataset(self)


def _build_dataset_registry() -> dict[str, PatchDataset]:
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
        if util.user_confirm("Display available datasets?"):
            print("\nAvailable datasets:\n-------------------")

            for dataset_name in _dataset_registry:
                print(dataset_name)

            print()

        name = input("Enter dataset name: ")

    dataset = _dataset_registry[name]
    dataset.set_version(version_name=version_name)

    return dataset
