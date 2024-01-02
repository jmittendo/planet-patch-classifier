from argparse import ArgumentParser, Namespace
from pathlib import Path

import source.satellite_dataset.adding as sd_adding


def main() -> None:
    input_args = parse_input_args()
    dataset_dir: str | None = input_args.path
    dataset_archive: str | None = input_args.archive
    dataset_name: str | None = input_args.name

    if dataset_dir is None:
        dataset_dir = input("Enter path: ")

    dataset_path = Path(dataset_dir)

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Directory '{dataset_path.as_posix()}' not found")

    if dataset_archive is None:
        dataset_archive = input("Enter dataset archive: ")

    if dataset_name is None:
        dataset_name = input("Enter dataset name: ")

    sd_adding.add_satellite_dataset(dataset_path, dataset_archive, dataset_name)


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description="Add a dataset at any path to the satellite datasets JSON-file.",
    )

    arg_parser.add_argument("path", nargs="?", help="path to the satellite dataset")

    arg_parser.add_argument(
        "archive", nargs="?", help="the archive this dataset originates from"
    )

    arg_parser.add_argument("name", nargs="?", help="name to give to the dataset")

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
