from pathlib import Path

from pandas import DataFrame
from planetaryimage import PDS3Image
from pvl import PVLModule, Quantity

import source.constants as constants
import source.dataset_validation as validation
import source.utility as util
from source.exceptions import ValidationError


def generate_satellite_dataset_table(
    dataset_archive: str, dataset_path: Path, output_path: Path
) -> None:
    is_valid, message = validation.validate_satellite_dataset(
        dataset_archive, dataset_path
    )

    if not is_valid:
        raise ValidationError(f"Dataset is invalid: {message}")

    match dataset_archive:
        case "vex-vmc":
            table = _generate_vex_vmc_table(dataset_path)
        case "vco":
            table = _generate_vco_table(dataset_path)
        case _:
            raise ValueError(
                "No table generation script implemented for dataset archive "
                f"'{dataset_archive}'"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_pickle(output_path)


def _generate_vex_vmc_table(dataset_path: Path) -> DataFrame:
    # Dataset should be validated before calling this function

    file_name_bases: list[str] = []
    img_file_paths: list[Path] = []
    geo_file_paths: list[Path] = []
    max_resolutions_mpx: list[float] = []

    for mission_dir_path in dataset_path.iterdir():
        img_file_dir_path = mission_dir_path / "DATA"
        geo_file_dir_path = mission_dir_path / "GEOMETRY"

        for img_file_orbit_dir_path in img_file_dir_path.iterdir():
            geo_file_orbit_dir_path = geo_file_dir_path / img_file_orbit_dir_path.name

            for img_file_path in img_file_orbit_dir_path.glob("*.IMG"):
                geo_file_path = (
                    geo_file_orbit_dir_path / img_file_path.with_suffix(".GEO").name
                )
                file_name_base = img_file_path.stem

                img_file = PDS3Image.open(img_file_path.as_posix())
                img_pds3_label: PVLModule = img_file.label  # type: ignore

                # Max resolution of a pixel in m/px
                max_resolution_quantity: Quantity = img_pds3_label["MAXIMUM_RESOLUTION"]  # type: ignore
                max_resolution_mpx: float = max_resolution_quantity.value

                file_name_bases.append(file_name_base)
                img_file_paths.append(img_file_path)
                geo_file_paths.append(geo_file_path)
                max_resolutions_mpx.append(max_resolution_mpx)

    table_dict = {
        "file_name_base": file_name_bases,
        "img_file_path": img_file_paths,
        "geo_file_path": geo_file_paths,
        "max_resolution_mpx": max_resolutions_mpx,
    }

    return DataFrame(data=table_dict)


def _generate_vco_table(dataset_path: Path) -> DataFrame:
    # Dataset should be validated before calling this function

    file_name_bases: list[str] = []
    img_file_paths: list[Path] = []
    geo_file_paths: list[Path] = []
    max_resolutions_mpx: list[float] = []

    geo_file_subdir_path = dataset_path / "extras"
    geo_file_dir_version_to_path_map = {
        path.name.split("_")[-1]: path for path in geo_file_subdir_path.iterdir()
    }

    for img_file_dir_path in dataset_path.iterdir():
        if img_file_dir_path.name == "extras":
            continue

        img_file_dir_version_str = img_file_dir_path.name.split("-")[-1]
        geo_file_dir_path = geo_file_dir_version_to_path_map[img_file_dir_version_str]

        for img_file_mission_dir_path in img_file_dir_path.iterdir():
            img_file_mission_dir_name_components = img_file_mission_dir_path.name.split(
                "_"
            )
            geo_file_mission_dir_name = (
                f"{img_file_mission_dir_name_components[0]}_7"
                f"{img_file_mission_dir_name_components[1][1:]}"
            )
            geo_file_mission_dir_path = geo_file_dir_path / geo_file_mission_dir_name

            img_file_level_dir_path = img_file_mission_dir_path / "data" / "l2b"
            geo_file_level_dir_path = (
                geo_file_mission_dir_path / "data" / "l3bx" / "fits"
            )

            for img_file_orbit_dir_path in img_file_level_dir_path.iterdir():
                geo_file_orbit_dir_path = (
                    geo_file_level_dir_path / img_file_orbit_dir_path.name
                )

                for img_file_path in img_file_orbit_dir_path.glob("*.fit"):
                    geo_file_path = (
                        geo_file_orbit_dir_path
                        / img_file_path.name.replace("l2b", "l3bx")
                    )
                    file_name_base = "_".join(img_file_path.name.split("_")[:4])

                    img_hdu = util.load_fits_hdu_or_hdus(img_file_path, 1)
                    img_header = img_hdu.header

                    # Distance between center of satellite and center of Venus:
                    # https://darts.isas.jaxa.jp/planet/project/akatsuki/doc/fits/vco_fits_dic_v07.html#s_distav
                    distance_km: float = img_header["S_DISTAV"]  # type: ignore

                    # FOV of a single pixel in radians:
                    # https://darts.isas.jaxa.jp/planet/project/akatsuki/doc/fits/vco_fits_dic_v07.html#s_ifov
                    pixel_fov_rad: float = img_header["S_IFOV"]  # type: ignore

                    altitude_m = distance_km * 1000 - constants.VENUS_RADIUS_M

                    # Approximation of max resolution of a pixel in m/px
                    max_resolution_mpx = pixel_fov_rad * altitude_m

                    file_name_bases.append(file_name_base)
                    img_file_paths.append(img_file_path)
                    geo_file_paths.append(geo_file_path)
                    max_resolutions_mpx.append(max_resolution_mpx)

    table_dict = {
        "file_name_base": file_name_bases,
        "img_file_path": img_file_paths,
        "geo_file_path": geo_file_paths,
        "max_resolution_mpx": max_resolutions_mpx,
    }

    return DataFrame(data=table_dict)
