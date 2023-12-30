from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
from pandas import DataFrame

import source.dataset_tables as tables
import source.utility as util
import user.config as config


def main():
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name
    regenerate_table: bool = input_args.regenerate_table

    dataset_name, dataset = util.load_satellite_dataset(dataset_name)
    dataset_archive = dataset["archive"]
    dataset_path = Path(dataset["path"])

    table_path = config.SATELLITE_DATASET_TABLES_DIR_PATH / f"{dataset_name}.pkl"

    if regenerate_table or not table_path.is_file():
        tables.generate_satellite_dataset_table(
            dataset_archive, dataset_path, table_path
        )

    dataset_table = pd.read_pickle(table_path)

    match dataset_archive:
        case "vex-vmc":
            generate_vex_vmc_patches(dataset_table)
        case "vco":
            generate_vco_patches(dataset_table)
        case _:
            raise ValueError(
                "No patch generation script implemented for dataset archive "
                f"'{dataset_archive}'"
            )


def generate_vex_vmc_patches(dataset_table: DataFrame):
    ...


def generate_vco_patches(dataset_table: DataFrame):
    ...


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=(
            "Generate patches for a named dataset. Run without arguments to get a list "
            "of available datasets."
        ),
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")
    arg_parser.add_argument(
        "-t",
        "--table",
        action="store_true",
        dest="regenerate_table",
        help="regenerate the dataset table file",
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
