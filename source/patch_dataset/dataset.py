import json
from pathlib import Path
from typing import Iterator, TypedDict

import pandas as pd
from pandas import DataFrame, Series

import source.patch_dataset.config as pd_config
import source.patch_dataset.plotting as pd_plotting
import source.utility as util


class _PatchDatasetInfoDict(TypedDict):
    scale_km: float
    resolution: int


class PatchDataset:
    def __init__(self, name: str, path: Path, satellite_dataset_name: str) -> None:
        self.name = name
        self.satellite_dataset_name = satellite_dataset_name

        versions_dir_path = path / "versions"
        self._version_dir_paths_dict = {
            p.name: p for p in versions_dir_path.iterdir() if p.is_dir()
        }

        table_path = path / "table.pkl"
        self._table: DataFrame = pd.read_pickle(table_path)

        dataset_info_json_path = path / "info.json"

        with open(dataset_info_json_path) as dataset_info_json:
            dataset_info_dict: _PatchDatasetInfoDict = json.load(dataset_info_json)

        self.scale_km = dataset_info_dict["scale_km"]
        self.resolution = dataset_info_dict["resolution"]

    def __iter__(self) -> Iterator[Series]:
        for index, data in self._table.iterrows():
            yield data

    def __len__(self) -> int:
        return len(self._table)

    def random_sample(self, num_patches: int) -> DataFrame:
        return self._table.sample(n=num_patches)

    def get_version_dir_path(self, version_name: str | None = None) -> Path:
        if version_name is None:
            if util.user_confirm("Display available versions?"):
                print("\nAvailable versions:\n-------------------")

                for version in self._version_dir_paths_dict:
                    print(version)

                print()

            version_name = input("Enter version name: ")

        return self._version_dir_paths_dict[version_name]

    def plot(
        self,
        version_name: str | None = None,
        num_patches: int | None = None,
    ) -> None:
        pd_plotting.plot_dataset(
            self, version_name=version_name, num_patches=num_patches
        )


def _build_dataset_registry() -> dict[str, PatchDataset]:
    dataset_registry: dict[str, PatchDataset] = {}

    for satellite_dataset_dir_path in pd_config.DATASETS_DIR_PATH.iterdir():
        if not satellite_dataset_dir_path.is_dir():
            continue

        for patch_dataset_path in satellite_dataset_dir_path.iterdir():
            if not patch_dataset_path.is_dir():
                continue

            dataset_name = patch_dataset_path.name
            dataset = PatchDataset(
                dataset_name, patch_dataset_path, satellite_dataset_dir_path.name
            )
            dataset_registry[dataset_name] = dataset

    return dataset_registry


_dataset_registry = _build_dataset_registry()


def get(name: str | None) -> PatchDataset:
    if name is None:
        if util.user_confirm("Display available datasets?"):
            print("\nAvailable datasets:\n-------------------")

            for dataset_name in _dataset_registry:
                print(dataset_name)

            print()

        name = input("Enter dataset name: ")

    return _dataset_registry[name]
