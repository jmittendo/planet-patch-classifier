from argparse import ArgumentParser, Namespace

import source.satellite_dataset as sd


def main() -> None:
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name

    dataset = sd.get_dataset(dataset_name)
    dataset.generate_table()


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=(
            "(Re)generate the table for a named dataset. Run without arguments to get "
            "a list of available datasets."
        ),
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
