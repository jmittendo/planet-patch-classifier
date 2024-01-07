import json
from json import JSONDecodeError
from pathlib import Path

import source.patch_dataset.config as pdcfg
import source.utility as util
import user.config as ucfg
from source.patch_dataset.typing import PatchDataset


def add_patch_dataset(dataset_path: Path, dataset_name: str) -> None:
    patch_datasets_json_path = pdcfg.DATASETS_JSON_PATH
    patch_datasets_json_path.parent.mkdir(parents=True, exist_ok=True)

    patch_datasets: dict[str, PatchDataset] = {}

    if patch_datasets_json_path.is_file():
        with open(patch_datasets_json_path, "r") as patch_datasets_json:
            try:
                patch_datasets = json.load(patch_datasets_json)
            except JSONDecodeError:
                print(
                    f"Warning: JSON file '{patch_datasets_json_path.as_posix()}' "
                    "contains invalid syntax and could not be deserialized."
                )

                if not util.user_confirm(
                    "Continue and overwrite existing file content?"
                ):
                    return

        if dataset_name in patch_datasets:
            print(
                f"Warning: '{patch_datasets_json_path.as_posix()}' already "
                f"contains a dataset with the name '{dataset_name}'."
            )

            if not util.user_confirm("Continue and overwrite existing dataset?"):
                return

    with open(patch_datasets_json_path, "w") as patch_datasets_json:
        patch_datasets[dataset_name] = {
            "name": dataset_name,
            "path": dataset_path.as_posix(),
        }

        json.dump(patch_datasets, patch_datasets_json, indent=ucfg.JSON_INDENT)

        print(
            f"Successfully added dataset '{dataset_name}' to "
            f"{patch_datasets_json_path.as_posix()}."
        )
