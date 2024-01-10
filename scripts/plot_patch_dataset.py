from argparse import ArgumentParser, Namespace

import source.patch_dataset.dataset as pd_dataset


def main() -> None:
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name
    patch_normalization: str | None = input_args.normalization
    num_patches: int | None = input_args.number

    dataset = pd_dataset.get(dataset_name)

    if patch_normalization is None:
        patch_normalization = input(
            "Enter patch normalization mode ('local', 'global'): "
        )

    dataset.plot(patch_normalization, num_patches=num_patches)  # type: ignore


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=("Generate a scatter plot for the specified dataset."),
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")
    arg_parser.add_argument(
        "normalization", nargs="?", help="patch normalization mode ('local', 'global')"
    )
    arg_parser.add_argument(
        "-n", "--number", type=int, help="number of patches to plot"
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
