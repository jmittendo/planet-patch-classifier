from argparse import ArgumentParser, Namespace

import source.satellite_dataset.dataset as sd_dataset


def main() -> None:
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name

    dataset = sd_dataset.load(dataset_name)

    print(f"\nNumber of files in the dataset: {len(dataset)}")


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=(
            "Count the number of files in a named dataset. Run without arguments to "
            "get a list of available datasets."
        ),
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
