import json
from json import JSONDecodeError
from pathlib import Path
from typing import Iterator, TypedDict

import pandas as pd
from pandas import DataFrame, Series

import source.satellite_dataset.archive as sd_archive
import source.satellite_dataset.config as sd_config
import source.utility as util
import user.config as user_config
from source.exceptions import ValidationError
from source.patch_dataset.typing import PatchNormalization
from source.satellite_dataset.archive import Archive


class _DatasetDict(TypedDict):
    name: str
    path: str
    archive: str


class Dataset:
    def __init__(self, name: str, path: Path, archive: Archive) -> None:
        self.name = name
        self.path = path
        self.archive = archive

        self.is_valid, validation_message = archive.validate_dataset(self)

        if not self.is_valid:
            raise ValidationError(f"Dataset is invalid: {validation_message}")

        self.table_path = sd_config.DATASET_TABLES_DIR_PATH / f"{self.name}.pkl"
        self.table = self._load_table()

    def __iter__(self) -> Iterator[Series]:
        for index, data in self.table.iterrows():
            yield data

    def __len__(self) -> int:
        return len(self.table)

    @classmethod
    def from_dict(cls, dataset_dict: _DatasetDict) -> "Dataset":
        name = dataset_dict["name"]
        path = Path(dataset_dict["path"])
        archive = sd_archive.get(dataset_dict["name"])

        return cls(name, path, archive)

    def generate_patches(
        self, scale_km: float, resolution: int, normalization: PatchNormalization
    ) -> None:
        self.archive.generate_dataset_patches(self, scale_km, resolution, normalization)

    def _load_table(self) -> DataFrame:
        if not self.table_path.is_file():
            self._generate_table()

        return pd.read_pickle(self.table_path)

    def _generate_table(self) -> None:
        table = self.archive.generate_dataset_table(self)

        self.table_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_pickle(self.table_path)


def _build_dataset_registry() -> dict[str, Dataset]:
    dataset_registry: dict[str, Dataset] = {}

    with open(sd_config.DATASETS_JSON_PATH) as datasets_json:
        dataset_dicts: list[_DatasetDict] = json.load(datasets_json)

        for dataset_dict in dataset_dicts:
            dataset_name = dataset_dict["name"]
            dataset_registry[dataset_name] = Dataset.from_dict(dataset_dict)

    return dataset_registry


_dataset_registry = _build_dataset_registry()


def get(name: str | None) -> Dataset:
    if name is None:
        if util.user_confirm("Display available datasets?"):
            print("\nAvailable datasets:\n-------------------")

            for dataset_name in _dataset_registry:
                print(dataset_name)

            print()

        name = input("Enter dataset name: ")

    return _dataset_registry[name]


def add(name: str, path: Path, archive: Archive) -> None:
    datasets_json_path = sd_config.DATASETS_JSON_PATH
    datasets_json_path.parent.mkdir(parents=True, exist_ok=True)

    datasets_dict: dict[str, _DatasetDict] = {}

    if datasets_json_path.is_file():
        with open(datasets_json_path, "r") as datasets_json:
            try:
                datasets_dict: dict[str, _DatasetDict] = json.load(datasets_json)
            except JSONDecodeError:
                print(
                    f"Warning: JSON file '{datasets_json_path.as_posix()}' "
                    "contains invalid syntax and could not be deserialized."
                )

                if not util.user_confirm(
                    "Continue and overwrite existing file content?"
                ):
                    return

        if name in datasets_dict:
            print(
                f"Warning: '{datasets_json_path.as_posix()}' already "
                f"contains a dataset with the name '{name}'."
            )

            if not util.user_confirm("Continue and overwrite existing dataset?"):
                return

    with open(datasets_json_path, "w") as datasets_json:
        datasets_dict[name] = {
            "name": name,
            "path": path.as_posix(),
            "archive": archive.name,
        }

        json.dump(datasets_dict, datasets_json, indent=user_config.JSON_INDENT)

        print(
            f"Successfully added dataset '{name}' to {datasets_json_path.as_posix()}."
        )
