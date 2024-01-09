import abc
import json
import typing
from abc import ABC
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TypedDict

import numpy as np
import spiceypy as spice
from pandas import DataFrame
from planetaryimage import PDS3Image
from pvl import PVLModule, Quantity

import source.satellite_dataset.config as sd_config
import source.satellite_dataset.planet as sd_planet
import source.satellite_dataset.utility as sd_util
import source.satellite_dataset.validation as sd_validation
from source.exceptions import ValidationError
from source.satellite_dataset.dataset import Dataset
from source.satellite_dataset.planet import Planet


class ArchiveType(StrEnum):
    IMG_GEO = "img-geo"
    IMG_SPICE = "img-spice"


class _ArchiveDict(TypedDict):
    name: str
    type: str
    planet: str
    spice_path: str | None


class Archive(ABC):
    def __init__(
        self, name: str, type: ArchiveType, planet: Planet, spice_path: Path | None
    ) -> None:
        self.name = name
        self.type = type
        self.planet = planet
        self.spice_path = spice_path

    @classmethod
    def from_dict(cls, archive_dict: _ArchiveDict) -> "Archive":
        name = archive_dict["name"]
        type = ArchiveType[archive_dict["type"]]
        planet = sd_planet.load(archive_dict["planet"])

        spice_path_str = archive_dict["spice_path"]
        spice_path = None if spice_path_str is None else Path(spice_path_str)

        return cls(name, type, planet, spice_path)

    @abc.abstractmethod
    def validate_dataset(self, dataset: Dataset) -> tuple[bool, str]:
        raise NotImplementedError

    @abc.abstractmethod
    def generate_dataset_table(self, dataset: Dataset) -> DataFrame:
        raise NotImplementedError


class VexVmcArchive(Archive):
    @typing.override
    def validate_dataset(self, dataset: Dataset) -> tuple[bool, str]:
        return sd_validation.validate_vex_vmc_dataset(dataset)

    @typing.override
    def generate_dataset_table(self, dataset: Dataset) -> DataFrame:
        if not dataset.is_valid:
            raise ValidationError("Dataset is invalid")

        if self.spice_path is None:
            raise ValueError(f"Spice path must not be 'null' for archive {self.name}")

        sd_util.load_spice_kernels(self.spice_path)

        file_name_bases: list[str] = []
        img_file_paths: list[Path] = []
        geo_file_paths: list[Path] = []
        max_resolutions_mpx: list[float] = []
        solar_longitudes_deg: list[float] = []

        for mission_dir_path in dataset.path.iterdir():
            img_file_dir_path = mission_dir_path / "DATA"
            geo_file_dir_path = mission_dir_path / "GEOMETRY"

            for img_file_orbit_dir_path in img_file_dir_path.iterdir():
                geo_file_orbit_dir_path = (
                    geo_file_dir_path / img_file_orbit_dir_path.name
                )

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
                    max_resolution_quantity: Quantity = img_pds3_label[  # type: ignore
                        "MAXIMUM_RESOLUTION"
                    ]
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
            "solar_longitude_deg": solar_longitudes_deg,
        }

        return DataFrame(data=table_dict)


class VcoArchive(Archive):
    @typing.override
    def validate_dataset(self, dataset: Dataset) -> tuple[bool, str]:
        return sd_validation.validate_vco_dataset(dataset)

    @typing.override
    def generate_dataset_table(self, dataset: Dataset) -> DataFrame:
        if not dataset.is_valid:
            raise ValidationError("Dataset is invalid")

        file_name_bases: list[str] = []
        img_file_paths: list[Path] = []
        geo_file_paths: list[Path] = []
        max_resolutions_mpx: list[float] = []
        solar_longitudes_deg: list[float] = []

        geo_file_subdir_path = dataset.path / "extras"
        geo_file_dir_version_to_path_map = {
            path.name.split("_")[-1]: path for path in geo_file_subdir_path.iterdir()
        }

        for img_file_dir_path in dataset.path.iterdir():
            if img_file_dir_path.name == "extras":
                continue

            img_file_dir_version_str = img_file_dir_path.name.split("-")[-1]
            geo_file_dir_path = geo_file_dir_version_to_path_map[
                img_file_dir_version_str
            ]

            for img_file_mission_dir_path in img_file_dir_path.iterdir():
                img_file_mission_dir_name_components = (
                    img_file_mission_dir_path.name.split("_")
                )
                geo_file_mission_dir_name = (
                    f"{img_file_mission_dir_name_components[0]}_7"
                    f"{img_file_mission_dir_name_components[1][1:]}"
                )
                geo_file_mission_dir_path = (
                    geo_file_dir_path / geo_file_mission_dir_name
                )

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
                        solar_longitude_deg = sd_util.fix_360_longitude(
                            solar_longitude_deg
                        )

                        altitude_m = (distance_km - self.planet.radius_km) * 1000

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


def load(name: str) -> Archive:
    with open(sd_config.ARCHIVES_JSON_PATH) as archives_json:
        archives_dict: dict[str, _ArchiveDict] = json.load(archives_json)

    return Archive.from_dict(archives_dict[name])
