import json
import logging
from datetime import datetime
from pathlib import Path

import user.config as config
from source.typing import SatelliteDataset


def load_satellite_dataset(dataset_name: str | None = None) -> SatelliteDataset:
    with open(config.SATELLITE_DATASETS_JSON_PATH, "r") as json_file:
        satellite_datasets: dict[str, SatelliteDataset] = json.load(json_file)

    if dataset_name is None:
        if user_confirm("Display available datasets?"):
            print("\nAvailable datasets:\n-------------------")

            for dataset_name in satellite_datasets:
                print(dataset_name)

            print()

        dataset_name = input("Enter dataset name: ")

    return satellite_datasets[dataset_name]


def user_confirm(message: str) -> bool:
    reply = input(f"{message} (y/n): ")

    while reply.lower() not in ("y", "n", "yes", "no"):
        reply = input("Please reply with 'yes'/'y' or 'no'/'n': ")

    return reply.lower() in ("y", "yes")


def configure_logging(log_file_name_base: str) -> None:
    current_datetime_str = datetime.now().strftime(config.DATETIME_FORMAT)

    config.LOGS_DIR_PATH.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=Path(
            config.LOGS_DIR_PATH, f"{log_file_name_base}_{current_datetime_str}.log"
        ),
        format=config.LOGGING_FORMAT,
        level=logging.DEBUG,
    )
