from argparse import ArgumentParser, Namespace

import source.satellite_dataset.patches as sd_patches
import source.satellite_dataset.utility as sd_util


def main() -> None:
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name
    patch_scale_km: float | None = input_args.scale
    patch_resolution: int | None = input_args.resolution
    regenerate_table: bool = input_args.table

    dataset_name, dataset = sd_util.load_dataset(dataset_name)

    if patch_scale_km is None:
        patch_scale_km = float(input("Enter scale of patches in km: "))

    if patch_resolution is None:
        patch_resolution = int(input("Enter pixel resolution of patches: "))

    sd_patches.generate_patches(
        dataset, dataset_name, patch_scale_km, patch_resolution, regenerate_table
    )


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=(
            "Generate patches for a named dataset. Run without arguments to get a list "
            "of available datasets."
        ),
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")
    arg_parser.add_argument(
        "scale", nargs="?", type=float, help="scale of the patches in km"
    )
    arg_parser.add_argument(
        "resolution", nargs="?", type=int, help="pixel resolution of the patches"
    )
    arg_parser.add_argument(
        "-t", "--table", action="store_true", help="regenerate the dataset table file"
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
