from argparse import ArgumentParser, Namespace

import source.patch_dataset.dataset as pd_dataset


def main() -> None:
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name
    subdir_name: str | None = input_args.subdir
    num_patches: int | None = input_args.number

    dataset = pd_dataset.get(dataset_name)
    subdirs_str = ", ".join(dataset.subdir_names)

    if subdir_name is None:
        subdir_name = input(
            f"Enter name of the dataset subdir to use for plotting ({subdirs_str}): "
        )

    dataset.plot(subdir_name, num_patches=num_patches)  # type: ignore


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=("Generate a scatter plot for the specified dataset."),
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")
    arg_parser.add_argument(
        "subdir", nargs="?", help="name of the dataset subdir to use for plotting"
    )
    arg_parser.add_argument(
        "-n", "--number", type=int, help="number of patches to plot"
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
