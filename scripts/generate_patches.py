from argparse import ArgumentParser, Namespace

import source.satellite_dataset.dataset as sd_dataset


def main() -> None:
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name
    patch_scale_km: float | None = input_args.scale
    patch_resolution: int | None = input_args.resolution
    patch_normalization: str | None = input_args.normalization

    dataset = sd_dataset.get(dataset_name)

    if patch_scale_km is None:
        patch_scale_km = float(input("Enter scale of patches in km: "))

    if patch_resolution is None:
        patch_resolution = int(input("Enter pixel resolution of patches: "))

    if patch_normalization is None:
        patch_normalization = input(
            "Enter patch normalization mode ('local', 'global', 'both'): "
        )

    dataset.generate_patches(patch_scale_km, patch_resolution, patch_normalization)  # type: ignore


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
        "normalization",
        nargs="?",
        help="patch normalization mode ('local', 'global', 'both')",
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
