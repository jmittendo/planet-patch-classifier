# This file is part of planet-patch-classifier, a Python tool for generating and
# classifying planet patches from satellite imagery via unsupervised machine learning
# Copyright (C) 2024  Jan Mittendorf (jmittendo)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
