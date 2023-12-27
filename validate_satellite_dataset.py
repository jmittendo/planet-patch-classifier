import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import source.utility as util
from source.typing import SatelliteDataset
from source.utility import config


def main() -> None:
    with open(config.satellite_datasets_json_path, "r") as json_file:
        satellite_datasets: dict[str, SatelliteDataset] = json.load(json_file)

    input_args = parse_input_args()
    dataset_name: str | None = input_args.name

    if dataset_name is None:
        if util.user_confirm("Display available datasets?"):
            print("\nAvailable datasets:\n-------------------")

            for dataset_name in satellite_datasets:
                print(dataset_name)

            print()

        dataset_name = input("Enter dataset name: ")

    dataset_archive = satellite_datasets[dataset_name]["archive"]
    dataset_path = Path(satellite_datasets[dataset_name]["path"])

    print()  # Empty line for better separation

    # Dataset path must be a valid directory
    if not dataset_path.is_dir():
        print(
            f"Dataset invalid: Dataset directory '{dataset_path.as_posix()}' not found"
        )
        return

    match dataset_archive:
        case "vex-vmc":
            validate_vex_vmc_dataset(dataset_path)
        case "vco":
            validate_vco_dataset(dataset_path)
        case _:
            raise ValueError(
                "No validation script implemented for dataset archive "
                f"'{dataset_archive}'"
            )


def validate_vex_vmc_dataset(dataset_path: Path) -> None:
    # Dataset directory must contain mission extension directories
    # (e.g. "VEX-V-VMC-3-RDR-V3.0" or "VEX-V-VMC-3-RDR-EXT1-V3.0", ...)
    mission_dir_paths = list(dataset_path.iterdir())

    if not mission_dir_paths:
        print("Dataset invalid: Dataset directory is empty")
        return

    for mission_dir_path in mission_dir_paths:
        if not mission_dir_path.is_dir():
            print(
                "Dataset invalid: Illegal file found in dataset directory: "
                f"'{mission_dir_path.as_posix()}'"
            )
            return

        # Mission extension directories must contain a directory with name "DATA" that
        # holds the image file orbit directories
        img_file_dir_path = mission_dir_path / "DATA"

        if not img_file_dir_path.is_dir():
            print(
                "Dataset invalid: Image file directory "
                f"'{img_file_dir_path.as_posix()}' not found"
            )
            return

        img_file_orbit_dir_paths = list(img_file_dir_path.iterdir())

        if not img_file_orbit_dir_paths:
            print(
                "Dataset invalid: Image file directory "
                f"'{img_file_dir_path.as_posix()}' is empty"
            )
            return

        for img_file_orbit_dir_path in img_file_orbit_dir_paths:
            # Image file orbit directories must contain image files with the extension
            # ".IMG"
            img_file_paths = list(img_file_orbit_dir_path.glob("*.IMG"))

            if not img_file_paths:
                print(
                    "Dataset invalid: Image file orbit directory "
                    f"'{img_file_orbit_dir_path.as_posix()}' does not contain '.IMG' "
                    "files"
                )
                return

            for img_file_path in img_file_paths:
                # Image files must have a corresponding geometry file with the following
                # name pattern and path
                geo_file_name = img_file_path.with_suffix(".GEO").name
                geo_file_path = (
                    mission_dir_path
                    / "GEOMETRY"
                    / img_file_orbit_dir_path.name
                    / geo_file_name
                )

                if not geo_file_path.is_file():
                    print(
                        f"Dataset invalid: Image file '{img_file_path.as_posix()}' "
                        "does not have a corresponding geometry file (Expected path: "
                        f"'{geo_file_path.as_posix()}')"
                    )
                    return

    print("Dataset valid: No errors found")


def validate_vco_dataset(dataset_path: Path) -> None:
    # Dataset directory must contain an "extras" subdir that holds the geometry file
    # directories
    geo_file_subdir_path = dataset_path / "extras"

    if not geo_file_subdir_path.is_dir():
        print(
            "Dataset invalid: Geometry file subdir "
            f"'{geo_file_subdir_path.as_posix()}' not found"
        )
        return

    geo_file_dir_paths = list(geo_file_subdir_path.iterdir())

    if not geo_file_dir_paths:
        print(
            "Dataset invalid: Geometry file subdir "
            f"'{geo_file_subdir_path.as_posix()}' is empty"
        )
        return

    # See explanation in next comment
    geo_file_dir_version_to_path_map: dict[str, Path] = {}

    for geo_file_dir_path in geo_file_dir_paths:
        if not geo_file_dir_path.is_dir():
            print(
                "Dataset invalid: Illegal file found in geometry file directory: "
                f"'{geo_file_dir_path.as_posix()}'"
            )
            return

        # Geometry file directories must have a version substring (e.g. "v1.0") at the
        # end of their name (separated by an underscore) to link them to their
        # corresponding image file directory later
        geo_file_dir_version_str = geo_file_dir_path.name.split("_")[-1]

        if not geo_file_dir_version_str.startswith("v"):
            print(
                "Dataset invalid: Geometry file directory "
                f"'{geo_file_dir_path.as_posix()}' does not have a valid version "
                "substring (e.g. 'v1.0') at the end of its name (separated by an "
                "underscore)"
            )
            return

        geo_file_dir_version_to_path_map[geo_file_dir_version_str] = geo_file_dir_path

    # Dataset directory must not be empty
    subdir_paths = list(dataset_path.iterdir())

    if not subdir_paths:
        print("Dataset invalid: Dataset directory is empty")
        return

    # Subdirs must either be image file directories (e.g. "vco-v-uvi-3-cdr-v1.0")
    # or the geometry file directory "extras"
    for subdir_path in subdir_paths:
        if not subdir_path.is_dir():
            print(
                "Dataset invalid: Illegal file found in dataset directory: "
                f"'{subdir_path.as_posix()}'"
            )
            return

        # Existence of geometry files is checked together with image files below
        if subdir_path.name == "extras":
            continue

        # Image file directories must have a version substring (e.g. "v1.0") at the end
        # of their name (separated by a dash) to link them to their corresponding
        # geometry file directory
        img_file_dir_version_str = subdir_path.name.split("-")[-1]

        if not img_file_dir_version_str.startswith("v"):
            print(
                f"Dataset invalid: Image file directory '{subdir_path.as_posix()}' "
                "does not have a valid version substring (e.g. 'v1.0') at the end of "
                "its name (separated by a dash)"
            )
            return

        # Image file directories must have a corresponding geometry file directory with
        # the same version substring
        geo_file_dir_path = geo_file_dir_version_to_path_map.get(
            img_file_dir_version_str
        )

        if geo_file_dir_path is None:
            print(
                f"Dataset invalid: Image file directory '{subdir_path.as_posix()}' "
                "does not have a corresponding geometry file directory (Expected "
                f"path: '{geo_file_dir_path.as_posix()}')"
            )
            return

        # Image file subdirs must contain mission extension directories
        # (e.g. "vcouvi_1001", "vcouvi_1002", ...)
        img_file_mission_dir_paths = list(subdir_path.iterdir())

        if not img_file_mission_dir_paths:
            print(
                f"Dataset invalid: Image file directory '{subdir_path.as_posix()}' is "
                "empty"
            )
            return

        for img_file_mission_dir_path in img_file_mission_dir_paths:
            if not img_file_mission_dir_path.is_dir():
                print(
                    "Dataset invalid: Illegal file found in image file directory: "
                    f"'{img_file_mission_dir_path.as_posix()}'"
                )
                return

            # Mission extension directories must contain a "/data/l2b/"" level directory
            img_file_level_dir_path = img_file_mission_dir_path / "data" / "l2b"

            if not img_file_level_dir_path.is_dir():
                print(
                    "Dataset invalid: Image file level directory "
                    f"'{img_file_level_dir_path.as_posix()}' not found"
                )
                return

            # Level directories must contain orbit directories
            # (e.g. "r0064", "r0065", ...)
            img_file_orbit_dir_paths = list(img_file_level_dir_path.iterdir())

            if not img_file_orbit_dir_paths:
                print(
                    "Dataset invalid: Image file level directory "
                    f"'{img_file_level_dir_path.as_posix()}' is empty"
                )
                return

            # The geometry file mission extension directory must have almost the same
            # name as the image file mission extension directory, but with a leading 7
            # in the number the underscore instead of a 1
            # (e.g. "vcouvi_1001" <-> "vcouvi_7001")
            img_file_mission_dir_name_components = img_file_mission_dir_path.name.split(
                "_"
            )
            geo_file_mission_dir_name = (
                f"{img_file_mission_dir_name_components[0]}_7"
                f"{img_file_mission_dir_name_components[1][1:]}"
            )
            geo_file_mission_dir_path = geo_file_dir_path / geo_file_mission_dir_name

            if not geo_file_mission_dir_path.is_dir():
                print(
                    "Dataset invalid: Geometry file mission extension directory "
                    f"'{geo_file_mission_dir_path.as_posix()}' not found"
                )
                return

            for img_file_orbit_dir_path in img_file_orbit_dir_paths:
                if not img_file_orbit_dir_path.is_dir():
                    print(
                        "Dataset invalid: Illegal file found in image file level "
                        f"directory: '{img_file_orbit_dir_path.as_posix()}'"
                    )
                    return

                # Orbit directories must contain .fit image files
                img_file_paths = list(img_file_orbit_dir_path.glob("*.fit"))

                if not img_file_paths:
                    print(
                        "Dataset invalid: Image file orbit directory "
                        f"'{img_file_orbit_dir_path.as_posix()}' does not contain "
                        "'.fit' files"
                    )
                    return

                # See explanation below
                img_file_name_base_set: set[str] = set()

                for img_file_path in img_file_paths:
                    # Image files must have a corresponding geometry file with the
                    # following name pattern and path
                    geo_file_name = img_file_path.name.replace("l2b", "l3bx")
                    geo_file_path = (
                        geo_file_mission_dir_path
                        / "data"
                        / "l3bx"
                        / "fits"
                        / img_file_orbit_dir_path.name
                        / geo_file_name
                    )

                    if not geo_file_path.is_file():
                        print(
                            f"Dataset invalid: Image file '{img_file_path.as_posix()}' "
                            "does not have a corresponding geometry file (Expected "
                            f"path: '{geo_file_path.as_posix()}')"
                        )
                        return

                    # Multiple version of an image file with the same file name base
                    # may exist in the online archive. Our final dataset must only
                    # contain one version (ideally the higher version) of said file
                    img_file_name_base = "_".join(img_file_path.stem.split("_")[:-1])

                    if img_file_name_base in img_file_name_base_set:
                        print(
                            f"Dataset invalid: Image file '{img_file_path.as_posix()}' "
                            "exists in at least two versions with file name base: "
                            f"'{img_file_name_base}'"
                        )
                        return

                    img_file_name_base_set.add(img_file_name_base)

    print("Dataset valid: No errors found")


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description="Valid the structure of a named dataset.",
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
