from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
from pandas import DataFrame

import source.satellite_dataset.table as sd_table
import source.satellite_dataset.utility as sd_util
import user.config as config
from source.satellite_dataset.typing import ImgGeoDataArrays


def main() -> None:
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name
    patch_scale_km: float | None = input_args.scale
    patch_resolution: int | None = input_args.resolution
    regenerate_table: bool = input_args.table

    dataset_name, dataset = sd_util.load_satellite_dataset(dataset_name)

    if patch_scale_km is None:
        patch_scale_km = float(input("Enter scale of patches in km: "))

    if patch_resolution is None:
        patch_resolution = int(input("Enter pixel resolution of patches: "))

    dataset_archive = dataset["archive"]
    dataset_path = Path(dataset["path"])

    table_path = config.SATELLITE_DATASET_TABLES_DIR_PATH / f"{dataset_name}.pkl"

    if regenerate_table or not table_path.is_file():
        sd_table.generate_satellite_dataset_table(
            dataset_archive, dataset_path, table_path
        )

    dataset_table = pd.read_pickle(table_path)

    # Spatial resolution of a patch in m/px (ignoring projection effects / distortions)
    patch_resolution_mpx = patch_scale_km * 1000 / patch_resolution

    match dataset_archive:
        case "vex-vmc" | "vco":
            generate_img_geo_patches(
                dataset_archive, dataset_table, patch_resolution_mpx
            )
        case _:
            raise ValueError(
                "No patch generation script implemented for dataset archive "
                f"'{dataset_archive}'"
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


def generate_img_geo_patches(
    dataset_archive: str, dataset_table: DataFrame, patch_resolution_mpx: float
) -> None:
    for row_index, row_data in dataset_table.iterrows():
        img_max_resolution_mpx: float = row_data["max_resolution_mpx"]

        if not passes_resolution_threshold(
            img_max_resolution_mpx, patch_resolution_mpx
        ):
            continue

        img_file_path = Path(row_data["img_file_path"])
        geo_file_path = Path(row_data["geo_file_path"])

        img_geo_data_arrays = load_img_geo_data_arrays(
            dataset_archive, img_file_path, geo_file_path
        )


def load_img_geo_data_arrays(
    dataset_archive: str, img_file_path: Path, geo_file_path: Path
) -> ImgGeoDataArrays:
    match dataset_archive:
        case "vex-vmc":
            raise NotImplementedError()
        case "vco":
            img_hdu = sd_util.load_fits_hdu_or_hdus(img_file_path, 1)
            lat_hdu, lon_hdu, lt_hdu, ina_hdu, ema_hdu = sd_util.load_fits_hdu_or_hdus(
                geo_file_path,
                [
                    "Latitude",
                    "Longitude",
                    "Local time",
                    "Incidence angle",
                    "Emission angle",
                ],
            )

            data_arrays: ImgGeoDataArrays = {
                "image": img_hdu.data,
                "latitude": lat_hdu.data,
                "longitude": lon_hdu.data,
                "local_time": lt_hdu.data,
                "incidence_angle": ina_hdu.data,
                "emission_angle": ema_hdu,
            }  # type: ignore

            return data_arrays
        case _:
            raise ValueError(
                "Can not load data arrays for unknown dataset archive "
                f"{dataset_archive}"
            )


def passes_resolution_threshold(
    img_max_resolution: float, patch_resolution: float
) -> bool:
    return img_max_resolution / patch_resolution < config.PATCH_RESOLUTION_TOLERANCE


if __name__ == "__main__":
    main()
