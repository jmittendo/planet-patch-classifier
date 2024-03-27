from argparse import ArgumentParser, Namespace

import source.satellite_dataset as sd


def main() -> None:
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name

    dataset = sd.get_dataset(dataset_name)
    is_valid, validation_message = dataset.validate()

    if is_valid:
        print("Dataset is valid")
    else:
        print(f"Dataset is invalid: {validation_message}")


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=(
            "Validate the structure of a satellite dataset. Run without arguments to "
            "get a list of available datasets."
        ),
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
