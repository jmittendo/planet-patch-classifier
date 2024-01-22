import json
from json import JSONDecodeError
from pathlib import Path
from typing import Iterator, TypedDict

import pandas as pd
from pandas import DataFrame, Series

import source.patch_dataset.config as pd_config
import source.patch_dataset.plotting as pd_plotting
import source.utility as util
import user.config as user_config
from source.patch_dataset.typing import PatchNormalization


class _PatchDatasetDict(TypedDict):
    name: str
    path: str
    satellite_dataset: str
    scale_km: float
    resolution: int


class PatchDataset:
    def __init__(
        self,
        name: str,
        path: Path,
        satellite_dataset_name: str,
        scale_km: float,
        resolution: int,
    ) -> None:
        self.name = name
        self.path = path
        self.satellite_dataset_name = satellite_dataset_name
        self.scale_km = scale_km
        self.resolution = resolution

        table_path = self.path / "patch-info.pkl"
        self._table: DataFrame = pd.read_pickle(table_path)

    def __iter__(self) -> Iterator[Series]:
        for index, data in self._table.iterrows():
            yield data

    def __len__(self) -> int:
        return len(self._table)

    @classmethod
    def from_dict(cls, dataset_dict: _PatchDatasetDict) -> "PatchDataset":
        name = dataset_dict["name"]
        path = Path(dataset_dict["path"])
        satellite_dataset_name = dataset_dict["satellite_dataset"]
        scale_km = dataset_dict["scale_km"]
        resolution = dataset_dict["resolution"]

        return cls(name, path, satellite_dataset_name, scale_km, resolution)

    def random_sample(self, num_patches: int) -> DataFrame:
        return self._table.sample(n=num_patches)

    def get_img_dir_path(self, patch_normalization: PatchNormalization) -> Path:
        img_dir_path = self.path / f"{patch_normalization}-normalization"

        if not img_dir_path.is_dir():
            raise FileNotFoundError(
                f"Could not find patch dataset directory '{img_dir_path.as_posix()}'"
            )

        return img_dir_path

    def plot(
        self,
        patch_normalization: PatchNormalization,
        num_patches: int | None = None,
    ) -> None:
        pd_plotting.plot_dataset(self, patch_normalization, num_patches=num_patches)


def _build_dataset_registry() -> dict[str, PatchDataset]:
    datasets_json_path = pd_config.DATASETS_JSON_PATH

    # If not patch dataset has been generated yet the file may not exist
    if not datasets_json_path.is_file():
        return {}

    dataset_registry: dict[str, PatchDataset] = {}

    with open(pd_config.DATASETS_JSON_PATH) as datasets_json:
        dataset_dicts: list[_PatchDatasetDict] = json.load(datasets_json)

        for dataset_dict in dataset_dicts:
            dataset_name = dataset_dict["name"]
            dataset_registry[dataset_name] = PatchDataset.from_dict(dataset_dict)

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


def add(
    name: str,
    path: Path,
    satellite_dataset_name: str,
    scale_km: float,
    resolution: int,
) -> None:
    datasets_json_path = pd_config.DATASETS_JSON_PATH
    datasets_json_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_dicts: list[_PatchDatasetDict] = []

    if datasets_json_path.is_file():
        with open(datasets_json_path, "r") as datasets_json:
            try:
                dataset_dicts: list[_PatchDatasetDict] = json.load(datasets_json)
            except JSONDecodeError:
                print(
                    f"Warning: JSON file '{datasets_json_path.as_posix()}' "
                    "contains invalid syntax and could not be deserialized."
                )

                if not util.user_confirm(
                    "Continue and overwrite existing file content?"
                ):
                    return

        dataset_names = [dataset_dict["name"] for dataset_dict in dataset_dicts]

        if name in dataset_names:
            print(
                f"Warning: '{datasets_json_path.as_posix()}' already "
                f"contains a dataset with the name '{name}'."
            )

            if not util.user_confirm("Continue and overwrite existing dataset?"):
                return

    with open(datasets_json_path, "w") as datasets_json:
        dataset_dicts.append(
            {
                "name": name,
                "path": path.as_posix(),
                "satellite_dataset": satellite_dataset_name,
                "scale_km": scale_km,
                "resolution": resolution,
            }
        )

        json.dump(dataset_dicts, datasets_json, indent=user_config.JSON_INDENT)

        print(
            f"Successfully added dataset '{name}' to {datasets_json_path.as_posix()}."
        )
