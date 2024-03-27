from argparse import ArgumentParser, Namespace

import source.satellite_dataset as sd


def main() -> None:
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name
    patch_scale_km: float | None = input_args.scale
    patch_resolution: int | None = input_args.resolution
    global_normalization: bool = input_args.globalnorm

    dataset = sd.get_dataset(dataset_name)

    if patch_scale_km is None:
        patch_scale_km = float(input("Enter scale of patches in km: "))

    if patch_resolution is None:
        patch_resolution = int(input("Enter pixel resolution of patches: "))

    dataset.generate_patches(
        patch_scale_km, patch_resolution, global_normalization=global_normalization
    )


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=(
            "Generate patches from a satellite dataset. Run without arguments to get a "
            "list of available datasets."
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
        "-g",
        "--globalnorm",
        action="store_true",
        help="enable additional global normalization output",
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
