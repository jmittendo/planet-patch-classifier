import json
from argparse import ArgumentParser, Namespace
from json import JSONDecodeError
from pathlib import Path

import source.utility as util
from source.utility import SatelliteDataset, config


def main() -> None:
    input_args = parse_input_args()
    dataset_dir: str | None = input_args.path
    satellite_name: str | None = input_args.satellite
    dataset_name: str | None = input_args.name

    if dataset_dir is None:
        dataset_dir = input("Enter path: ")

    dataset_path = Path(dataset_dir)

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Directory '{dataset_path.as_posix()}' not found")

    if satellite_name is None:
        satellite_name = input("Enter satellite name: ")

    if dataset_name is None:
        dataset_name = input("Enter dataset name: ")

    add_satellite_dataset(dataset_path, satellite_name, dataset_name)


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description="Add a dataset at any path to the satellite datasets JSON-file.",
    )

    arg_parser.add_argument("path", nargs="?", help="path to the satellite dataset")

    arg_parser.add_argument(
        "satellite", nargs="?", help="the satellite this dataset originates from"
    )

    arg_parser.add_argument("name", nargs="?", help="name to give to the dataset")

    return arg_parser.parse_args()


def add_satellite_dataset(
    dataset_path: Path, satellite_name: str, dataset_name: str
) -> None:
    satellite_datasets_json_path = config.satellite_datasets_json_path
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
            "satellite": satellite_name,
        }

        json.dump(satellite_datasets, json_file, indent=config.json_indent)

        print(
            f"Successfully added dataset '{dataset_name}' to "
            f"{satellite_datasets_json_path.as_posix()}."
        )


if __name__ == "__main__":
    main()
