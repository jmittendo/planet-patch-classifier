from pathlib import Path

import numpy as np
import spiceypy as spice
from pandas import DataFrame
from planetaryimage import PDS3Image
from pvl import PVLModule, Quantity

import source.satellite_dataset.config as sdcfg
import source.satellite_dataset.utility as sd_util
import source.satellite_dataset.validation as sd_validation
from source.exceptions import ValidationError
from source.satellite_dataset.typing import SatelliteDataArchive, SatelliteDataset


def generate_dataset_table(
    dataset: SatelliteDataset, archive: SatelliteDataArchive, output_path: Path
) -> None:
    is_valid, message = sd_validation.validate_dataset(dataset)

    if not is_valid:
        raise ValidationError(f"Dataset is invalid: {message}")

    dataset_path = Path(dataset["path"])
    archive_name = archive["name"]

    match archive_name:
        case "vex-vmc":
            spice_kernels_path_str = archive["spice"]

            if spice_kernels_path_str is None:
                raise ValueError(
                    f"Spice value must not be 'null' for archive {archive_name}"
                )

            spice_kernels_path = Path(spice_kernels_path_str)
            table = _generate_vex_vmc_table(dataset_path, spice_kernels_path)
        case "vco":
            table = _generate_vco_table(dataset_path)
        case "juno-jnc":
            raise NotImplementedError
        case _:
            raise ValueError(
                "No table generation script implemented for archive "
                f"'{archive_name}'"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_pickle(output_path)


def _generate_vex_vmc_table(dataset_path: Path, spice_kernels_path: Path) -> DataFrame:
    # Dataset should be validated before calling this function

    sd_util.load_spice_kernels(spice_kernels_path)

    file_name_bases: list[str] = []
    img_file_paths: list[Path] = []
    geo_file_paths: list[Path] = []
    max_resolutions_mpx: list[float] = []
    solar_longitudes_deg: list[float] = []

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

                # Venus Express VMC label keywords:
                # https://archives.esac.esa.int/psa/ftp/VENUS-EXPRESS/VMC/VEX-V-VMC-3-RDR-V3.0/DOCUMENT/VMC_EAICD.PDF

                # Max resolution of a pixel in m/px
                max_resolution_quantity: Quantity = img_pds3_label["MAXIMUM_RESOLUTION"]  # type: ignore
                max_resolution_mpx: float = max_resolution_quantity.value

                # Date and time of middle of image acquisition in UTC format
                # "YYYY-MM-DDTHH:MM:SS.MMMZ"
                img_datetime: datetime = img_pds3_label["IMAGE_TIME"]  # type: ignore

                img_time_str = img_datetime.strftime(r"%Y-%m-%d %H:%M:%S.%f")
                img_time_et: float = spice.str2et(img_time_str)  # type: ignore

                # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/spicelib/subslr.html
                sub_solar_point = spice.subslr(
                    "INTERCEPT/ELLIPSOID",
                    "VENUS",
                    img_time_et,
                    "IAU_VENUS",
                    "LT+S",
                    "VENUS",
                )[0]

                # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/spicelib/recsph.html
                solar_longitude_rad = spice.recsph(sub_solar_point)[2]
                solar_longitude_deg = np.rad2deg(solar_longitude_rad)

                file_name_bases.append(file_name_base)
                img_file_paths.append(img_file_path)
                geo_file_paths.append(geo_file_path)
                max_resolutions_mpx.append(max_resolution_mpx)
                solar_longitudes_deg.append(solar_longitude_deg)

    table_dict = {
        "file_name_base": file_name_bases,
        "img_file_path": img_file_paths,
        "geo_file_path": geo_file_paths,
        "max_resolution_mpx": max_resolutions_mpx,
        "solar_longitudes_deg": solar_longitudes_deg,
    }

    return DataFrame(data=table_dict)


def _generate_vco_table(dataset_path: Path) -> DataFrame:
    # Dataset should be validated before calling this function

    file_name_bases: list[str] = []
    img_file_paths: list[Path] = []
    geo_file_paths: list[Path] = []
    max_resolutions_mpx: list[float] = []
    solar_longitudes_deg: list[float] = []

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

                    img_header = sd_util.load_fits_header(img_file_path, 1)

                    # Akatsuki header keywords:
                    # https://darts.isas.jaxa.jp/planet/project/akatsuki/doc/l3/vco_l3_variable_v01.pdf

                    # Distance between center of satellite and center of Venus
                    distance_km: float = img_header["S_DISTAV"]  # type: ignore

                    # FOV of a single pixel in radians
                    pixel_fov_rad: float = img_header["S_IFOV"]  # type: ignore

                    # Planetocentric longitude of the sun in degrees east
                    solar_longitude_deg: float = img_header["S_SOLLON"]  # type: ignore

                    altitude_m = distance_km * 1000 - sdcfg.VENUS_RADIUS_M

                    # Approximation of max resolution of a pixel in m/px
                    max_resolution_mpx = pixel_fov_rad * altitude_m

                    file_name_bases.append(file_name_base)
                    img_file_paths.append(img_file_path)
                    geo_file_paths.append(geo_file_path)
                    max_resolutions_mpx.append(max_resolution_mpx)
                    solar_longitudes_deg.append(solar_longitude_deg)

    table_dict = {
        "file_name_base": file_name_bases,
        "img_file_path": img_file_paths,
        "geo_file_path": geo_file_paths,
        "max_resolution_mpx": max_resolutions_mpx,
        "solar_longitude_deg": solar_longitudes_deg,
    }

    return DataFrame(data=table_dict)
