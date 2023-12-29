from pathlib import Path


def validate_satellite_dataset(dataset_archive: str, dataset_path: Path):
    # Dataset path must be a valid directory
    if not dataset_path.is_dir():
        print(
            f"Dataset invalid: Dataset directory '{dataset_path.as_posix()}' not found"
        )
        return

    match dataset_archive:
        case "vex-vmc":
            _validate_vex_vmc_dataset(dataset_path)
        case "vco":
            _validate_vco_dataset(dataset_path)
        case _:
            raise ValueError(
                "No validation script implemented for dataset archive "
                f"'{dataset_archive}'"
            )


def _validate_vex_vmc_dataset(dataset_path: Path) -> bool:
    # Dataset directory must contain mission extension directories
    # (e.g. "VEX-V-VMC-3-RDR-V3.0" or "VEX-V-VMC-3-RDR-EXT1-V3.0", ...)
    mission_dir_paths = list(dataset_path.iterdir())

    if not mission_dir_paths:
        print("Dataset invalid: Dataset directory is empty")
        return False

    for mission_dir_path in mission_dir_paths:
        if not mission_dir_path.is_dir():
            print(
                "Dataset invalid: Illegal file found in dataset directory: "
                f"'{mission_dir_path.as_posix()}'"
            )
            return False

        # Mission extension directories must contain a directory with name "DATA" that
        # holds the image file orbit directories
        img_file_dir_path = mission_dir_path / "DATA"

        if not img_file_dir_path.is_dir():
            print(
                "Dataset invalid: Image file directory "
                f"'{img_file_dir_path.as_posix()}' not found"
            )
            return False

        img_file_orbit_dir_paths = list(img_file_dir_path.iterdir())

        if not img_file_orbit_dir_paths:
            print(
                "Dataset invalid: Image file directory "
                f"'{img_file_dir_path.as_posix()}' is empty"
            )
            return False

        for img_file_orbit_dir_path in img_file_orbit_dir_paths:
            # Image file orbit directories must contain image files with the extension
            # ".IMG" (but can also be empty!)
            for img_file_path in img_file_orbit_dir_path.glob("*.IMG"):
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
                    return False

    print("Dataset valid: No errors found")
    return True


def _validate_vco_dataset(dataset_path: Path) -> bool:
    # Dataset directory must contain an "extras" subdir that holds the geometry file
    # directories
    geo_file_subdir_path = dataset_path / "extras"

    if not geo_file_subdir_path.is_dir():
        print(
            "Dataset invalid: Geometry file subdir "
            f"'{geo_file_subdir_path.as_posix()}' not found"
        )
        return False

    geo_file_dir_paths = list(geo_file_subdir_path.iterdir())

    if not geo_file_dir_paths:
        print(
            "Dataset invalid: Geometry file subdir "
            f"'{geo_file_subdir_path.as_posix()}' is empty"
        )
        return False

    # See explanation in next comment
    geo_file_dir_version_to_path_map: dict[str, Path] = {}

    for geo_file_dir_path in geo_file_dir_paths:
        if not geo_file_dir_path.is_dir():
            print(
                "Dataset invalid: Illegal file found in geometry file directory: "
                f"'{geo_file_dir_path.as_posix()}'"
            )
            return False

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
            return False

        geo_file_dir_version_to_path_map[geo_file_dir_version_str] = geo_file_dir_path

    # Dataset directory must not be empty
    subdir_paths = list(dataset_path.iterdir())

    if not subdir_paths:
        print("Dataset invalid: Dataset directory is empty")
        return False

    # Subdirs must either be image file directories (e.g. "vco-v-uvi-3-cdr-v1.0")
    # or the geometry file directory "extras"
    for subdir_path in subdir_paths:
        if not subdir_path.is_dir():
            print(
                "Dataset invalid: Illegal file found in dataset directory: "
                f"'{subdir_path.as_posix()}'"
            )
            return False

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
            return False

        # Image file directories must have a corresponding geometry file directory with
        # the same version substring
        geo_file_dir_path = geo_file_dir_version_to_path_map.get(
            img_file_dir_version_str
        )

        if geo_file_dir_path is None:
            print(
                f"Dataset invalid: Image file directory '{subdir_path.as_posix()}' "
                "does not have a corresponding geometry file directory"
            )
            return False

        # Image file subdirs must contain mission extension directories
        # (e.g. "vcouvi_1001", "vcouvi_1002", ...)
        img_file_mission_dir_paths = list(subdir_path.iterdir())

        if not img_file_mission_dir_paths:
            print(
                f"Dataset invalid: Image file directory '{subdir_path.as_posix()}' is "
                "empty"
            )
            return False

        for img_file_mission_dir_path in img_file_mission_dir_paths:
            if not img_file_mission_dir_path.is_dir():
                print(
                    "Dataset invalid: Illegal file found in image file directory: "
                    f"'{img_file_mission_dir_path.as_posix()}'"
                )
                return False

            # Mission extension directories must contain a "/data/l2b/"" level directory
            img_file_level_dir_path = img_file_mission_dir_path / "data" / "l2b"

            if not img_file_level_dir_path.is_dir():
                print(
                    "Dataset invalid: Image file level directory "
                    f"'{img_file_level_dir_path.as_posix()}' not found"
                )
                return False

            # Level directories must contain orbit directories
            # (e.g. "r0064", "r0065", ...)
            img_file_orbit_dir_paths = list(img_file_level_dir_path.iterdir())

            if not img_file_orbit_dir_paths:
                print(
                    "Dataset invalid: Image file level directory "
                    f"'{img_file_level_dir_path.as_posix()}' is empty"
                )
                return False

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
                return False

            for img_file_orbit_dir_path in img_file_orbit_dir_paths:
                if not img_file_orbit_dir_path.is_dir():
                    print(
                        "Dataset invalid: Illegal file found in image file level "
                        f"directory: '{img_file_orbit_dir_path.as_posix()}'"
                    )
                    return False

                # See explanation below
                img_file_name_base_set: set[str] = set()

                # Orbit directories must contain .fit image files
                # (but can also be empty!)
                for img_file_path in img_file_orbit_dir_path.glob("*.fit"):
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
                        return False

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
                        return False

                    img_file_name_base_set.add(img_file_name_base)

    print("Dataset valid: No errors found")
    return True