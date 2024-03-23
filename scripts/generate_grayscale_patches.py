from argparse import ArgumentParser, Namespace

import source.patch_dataset as pd


def main() -> None:
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name
    version_name: str | None = input_args.version

    dataset = pd.get_dataset(name=dataset_name, version_name=version_name)

    pd.generate_grayscale_version(dataset)


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=("Generate a grayscale version of the specified dataset."),
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")
    arg_parser.add_argument("version", nargs="?", help="version of the dataset")

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
