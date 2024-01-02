import json


import source.satellite_dataset.config as sdcfg
import source.utility as util
from source.satellite_dataset.typing import SatelliteDataset


def load_dataset(dataset_name: str | None = None) -> tuple[str, SatelliteDataset]:
    with open(sdcfg.DATASETS_JSON_PATH, "r") as json_file:
        satellite_datasets: dict[str, SatelliteDataset] = json.load(json_file)

    if dataset_name is None:
        if util.user_confirm("Display available datasets?"):
            print("\nAvailable datasets:\n-------------------")

            for dataset_name in satellite_datasets:
                print(dataset_name)

            print()

        dataset_name = input("Enter dataset name: ")

    return dataset_name, satellite_datasets[dataset_name]
