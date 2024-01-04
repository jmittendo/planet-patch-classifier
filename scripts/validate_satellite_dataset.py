from argparse import ArgumentParser, Namespace

import source.satellite_dataset.utility as sd_util
import source.satellite_dataset.validation as sd_validation


def main() -> None:
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name

    dataset = sd_util.load_dataset(dataset_name=dataset_name)

    print()  # Empty line for better separation

    is_valid, message = sd_validation.validate_dataset(dataset)

    if is_valid:
        print(f"Dataset is valid: {message}")
    else:
        print(f"Dataset is invalid: {message}")


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
