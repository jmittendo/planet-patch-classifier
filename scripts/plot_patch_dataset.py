from argparse import ArgumentParser, Namespace

import source.patch_dataset.dataset as pd_dataset


def main() -> None:
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name
    version_name: str | None = input_args.version
    num_patches: int | None = input_args.number

    dataset = pd_dataset.get(dataset_name)
    dataset.plot(version_name=version_name, num_patches=num_patches)


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=("Generate a scatter plot for the specified dataset."),
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")
    arg_parser.add_argument(
        "version", nargs="?", help="version of the dataset to use for plotting"
    )
    arg_parser.add_argument(
        "-n", "--number", type=int, help="number of patches to plot"
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
