from argparse import ArgumentParser, Namespace
from pathlib import Path

import source.utility as util


def main():
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name

    dataset = util.load_satellite_dataset(dataset_name=dataset_name)
    dataset_archive = dataset["archive"]
    dataset_path = Path(dataset["path"])

    match dataset_archive:
        case "vex-vmc":
            generate_vex_vmc_patches(dataset_path)
        case "vco":
            generate_vco_patches(dataset_path)
        case _:
            raise ValueError(
                "No patch generation script implemented for dataset archive "
                f"'{dataset_archive}'"
            )


def generate_vex_vmc_patches(dataset_path: Path):
    ...


def generate_vco_patches(dataset_path: Path):
    ...


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=(
            "Generate patches for a named dataset. Run without arguments to get a list "
            "of available datasets."
        ),
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
