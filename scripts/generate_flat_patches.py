from argparse import ArgumentParser, Namespace

import source.patch_dataset as pd


def main() -> None:
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name
    blur_sigma: float | None = input_args.sigma
    contrast: float | None = input_args.contrast

    dataset = pd.get_dataset(name=dataset_name, version_name="norm-local")

    if blur_sigma is None:
        blur_sigma = float(input("Enter Gaussian blur radius: "))

    if contrast is None:
        contrast = float(input("Enter contrast: "))

    pd.generate_flat_version(dataset, blur_sigma, contrast)


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=("Generate a flattened version of the specified patch dataset."),
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")
    arg_parser.add_argument(
        "sigma", nargs="?", type=float, help="sigma value for the Gaussian blur"
    )
    arg_parser.add_argument(
        "contrast", nargs="?", type=float, help="contrast for standardization"
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
