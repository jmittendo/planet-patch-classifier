import json

import source.patch_dataset.config as pdcfg
import source.utility as util
from source.patch_dataset.typing import PatchDataset


def load_dataset(dataset_name: str | None = None) -> PatchDataset:
    with open(pdcfg.DATASETS_JSON_PATH, "r") as datasets_json:
        patch_datasets: dict[str, PatchDataset] = json.load(datasets_json)

    if dataset_name is None:
        if util.user_confirm("Display available datasets?"):
            print("\nAvailable datasets:\n-------------------")

            for dataset_name in patch_datasets:
                print(dataset_name)

            print()

        dataset_name = input("Enter dataset name: ")

    return patch_datasets[dataset_name]
