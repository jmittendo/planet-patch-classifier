import json
from json import JSONDecodeError
from pathlib import Path
from typing import TypedDict

import pandas as pd
from pandas import DataFrame

import source.satellite_dataset.archive as sd_archive
import source.satellite_dataset.config as sd_config
import source.satellite_dataset.table as sd_table
import source.satellite_dataset.validation as sd_validation
import source.utility as util
import user.config as user_config
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

        sd_validation.validate_dataset(self)

        table_path = sd_config.DATASET_TABLES_DIR_PATH / f"{self.name}.pkl"
        self.table = self._load_table(table_path)

        self.num_files = len(self.table)

    @classmethod
    def from_dict(cls, dataset_dict: _DatasetDict) -> "Dataset":
        name = dataset_dict["name"]
        path = Path(dataset_dict["path"])
        archive = sd_archive.load(dataset_dict["name"])

        return cls(name, path, archive)

    def _load_table(self, path: Path) -> DataFrame:
        if not path.is_file():
            self._generate_table(path)

        return pd.read_pickle(path)

    def _generate_table(self, path: Path) -> None:
        sd_table.generate_dataset_table(self, path)


def load(name: str | None) -> Dataset:
    with open(sd_config.DATASETS_JSON_PATH) as datasets_json:
        datasets_dict: dict[str, _DatasetDict] = json.load(datasets_json)

    if name is None:
        if util.user_confirm("Display available datasets?"):
            print("\nAvailable datasets:\n-------------------")

            for dataset_name in datasets_dict:
                print(dataset_name)

            print()

        name = input("Enter dataset name: ")

    return Dataset.from_dict(datasets_dict[name])


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
