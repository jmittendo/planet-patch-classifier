import json
from json import JSONDecodeError
from pathlib import Path

import source.satellite_dataset.config as sdcfg
import source.utility as util
import user.config as ucfg
from source.satellite_dataset.typing import SatelliteDataset


def add_satellite_dataset(
    dataset_path: Path, dataset_archive: str, dataset_name: str
) -> None:
    satellite_datasets_json_path = sdcfg.DATASETS_JSON_PATH
    satellite_datasets_json_path.parent.mkdir(parents=True, exist_ok=True)

    satellite_datasets: dict[str, SatelliteDataset] = {}

    if satellite_datasets_json_path.is_file():
        with open(satellite_datasets_json_path, "r") as json_file:
            try:
                satellite_datasets = json.load(json_file)
            except JSONDecodeError:
                print(
                    f"Warning: JSON file '{satellite_datasets_json_path.as_posix()}' "
                    "contains invalid syntax and could not be deserialized."
                )

                if not util.user_confirm(
                    "Continue and overwrite existing file content?"
                ):
                    return

        if dataset_name in satellite_datasets:
            print(
                f"Warning: '{satellite_datasets_json_path.as_posix()}' already "
                f"contains a dataset with the name '{dataset_name}'."
            )

            if not util.user_confirm("Continue and overwrite existing dataset?"):
                return

    with open(satellite_datasets_json_path, "w") as json_file:
        satellite_datasets[dataset_name] = {
            "path": dataset_path.as_posix(),
            "archive": dataset_archive,
        }

        json.dump(satellite_datasets, json_file, indent=ucfg.JSON_INDENT)

        print(
            f"Successfully added dataset '{dataset_name}' to "
            f"{satellite_datasets_json_path.as_posix()}."
        )
