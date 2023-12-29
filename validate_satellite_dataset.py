from argparse import ArgumentParser, Namespace
from pathlib import Path

import source.dataset_validation as validation
import source.utility as util


def main() -> None:
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name

    dataset = util.load_satellite_dataset(dataset_name=dataset_name)
    dataset_archive = dataset["archive"]
    dataset_path = Path(dataset["path"])

    print()  # Empty line for better separation
    validation.validate_satellite_dataset(dataset_archive, dataset_path)


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=(
            "Validate the structure of a named dataset. Run without arguments to get a "
            "list of available datasets."
        ),
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
