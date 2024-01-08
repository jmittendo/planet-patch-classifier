from argparse import ArgumentParser, Namespace

import source.satellite_dataset.counting as sd_counting
import source.satellite_dataset.utility as sd_util


def main() -> None:
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name
    regenerate_table: bool = input_args.table

    dataset = sd_util.load_dataset(dataset_name)

    print()  # Empty line for better separation

    sd_counting.count_dataset_files(dataset, regenerate_table)


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=(
            "Count the number of files in a named dataset. Run without arguments to "
            "get a list of available datasets."
        ),
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")
    arg_parser.add_argument(
        "-t", "--table", action="store_true", help="regenerate the dataset table file"
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
