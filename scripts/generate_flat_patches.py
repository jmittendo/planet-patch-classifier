from argparse import ArgumentParser, Namespace

import source.patch_dataset.dataset as pd_dataset
import source.patch_dataset.flattening as pd_flattening


def main() -> None:
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name
    blur_sigma: float | None = input_args.sigma

    dataset = pd_dataset.get(name=dataset_name, version_name="norm-local")

    if blur_sigma is None:
        blur_sigma = float(input("Enter Gaussian blur radius: "))

    pd_flattening.generate_flat_dataset(dataset, blur_sigma)


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=("Generate a flattened version of the specified dataset."),
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")
    arg_parser.add_argument(
        "-s", "--sigma", type=float, help="sigma value for the Gaussian blur"
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
