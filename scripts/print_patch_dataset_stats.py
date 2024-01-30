from argparse import ArgumentParser, Namespace

import source.patch_dataset.dataset as pd_dataset


def main() -> None:
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name
    version_name: str | None = input_args.version

    dataset = pd_dataset.get(name=dataset_name, version_name=version_name)
    dataset.load_images()

    print(f"Number of patches: {len(dataset)}")
    print(f"Channel means: {dataset.mean.tolist()}")
    print(f"Channel standard deviations: {dataset.std.tolist()}")


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=("Print some stats for the specified dataset."),
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")
    arg_parser.add_argument("version", nargs="?", help="version of the dataset")

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
