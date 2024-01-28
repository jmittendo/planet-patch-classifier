import abc
import typing
import warnings
from abc import ABC
from pathlib import Path

from pandas import DataFrame

import source.satellite_dataset.patches as sd_patches
import source.satellite_dataset.planet as sd_planet
import source.satellite_dataset.table as sd_table
import source.satellite_dataset.validation as sd_validation
from source.satellite_dataset.planet import Planet
from source.satellite_dataset.typing import ImgGeoDataArrays

if typing.TYPE_CHECKING:
    from source.satellite_dataset.dataset import SatelliteDataset


class Archive(ABC):
    _subclass_registry: dict[str, type["Archive"]] = {}

    def __init__(self, name: str, path: Path, planet: Planet) -> None:
        self.name = name
        self.planet = planet
        self.datasets_path = path / "datasets"

        spice_path: Path | None = path / "spice-kernels"

        if not spice_path.is_dir():
            spice_path = None

        self.spice_path = spice_path

    @classmethod
    def create(cls, path: Path, planet: Planet) -> "Archive":
        archive_name = path.name
        archive_subclass = cls._subclass_registry[archive_name]

        return archive_subclass(archive_name, path, planet)

    @abc.abstractmethod
    def validate_dataset(self, dataset: "SatelliteDataset") -> tuple[bool, str]:
        raise NotImplementedError

    @abc.abstractmethod
    def generate_dataset_table(self, dataset: "SatelliteDataset") -> DataFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def generate_dataset_patches(
        self,
        dataset: "SatelliteDataset",
        patch_scale_km: float,
        patch_resolution: int,
        global_normalization: bool = False,
    ) -> None:
        raise NotImplementedError


class ImgGeoArchive(Archive):
    def __init_subclass__(cls, name: str) -> None:
        cls._subclass_registry[name] = cls

    def generate_dataset_patches(
        self,
        dataset: "SatelliteDataset",
        patch_scale_km: float,
        patch_resolution: int,
        global_normalization: bool = False,
    ) -> None:
        sd_patches.generate_img_geo_patches(
            self,
            dataset,
            patch_scale_km,
            patch_resolution,
            global_normalization=global_normalization,
        )

    @abc.abstractmethod
    def load_data_arrays(
        self, img_file_path: Path, geo_file_path: Path
    ) -> ImgGeoDataArrays:
        raise NotImplementedError


class ImgSpiceArchive(Archive):
    def __init_subclass__(cls, name: str) -> None:
        cls._subclass_registry[name] = cls

    def generate_dataset_patches(
        self,
        dataset: "SatelliteDataset",
        patch_scale_km: float,
        patch_resolution: int,
        global_normalization: bool = False,
    ) -> None:
        warnings.warn(
            "'generate_dataset_patches' method not yet implemented for "
            "'ImgSpiceArchive'"
        )


class VexVmcArchive(ImgGeoArchive, name="vex-vmc"):
    def validate_dataset(self, dataset: "SatelliteDataset") -> tuple[bool, str]:
        return sd_validation.validate_vex_vmc_dataset(dataset)

    def generate_dataset_table(self, dataset: "SatelliteDataset") -> DataFrame:
        if self.spice_path is None:
            raise ValueError(
                f"Spice kernel directory not found for archive '{self.name}'"
            )

        return sd_table.generate_vex_vmc_dataset_table(dataset, self.spice_path)

    def load_data_arrays(
        self, img_file_path: Path, geo_file_path: Path
    ) -> ImgGeoDataArrays:
        return sd_patches.load_vex_vmc_data_arrays(img_file_path, geo_file_path)


class VcoArchive(ImgGeoArchive, name="vco"):
    def validate_dataset(self, dataset: "SatelliteDataset") -> tuple[bool, str]:
        return sd_validation.validate_vco_dataset(dataset)

    def generate_dataset_table(self, dataset: "SatelliteDataset") -> DataFrame:
        return sd_table.generate_vco_dataset_table(dataset, self.planet.radius_km)

    def load_data_arrays(
        self, img_file_path: Path, geo_file_path: Path
    ) -> ImgGeoDataArrays:
        return sd_patches.load_vco_data_arrays(img_file_path, geo_file_path)


class JunoJncArchive(ImgSpiceArchive, name="juno-jnc"):
    def validate_dataset(self, dataset: "SatelliteDataset") -> tuple[bool, str]:
        warnings.warn(
            "'validate_dataset' method not yet implemented for 'ImgSpiceArchive'"
        )
        return True, ""  # TEMPORARY

    def generate_dataset_table(self, dataset: "SatelliteDataset") -> DataFrame:
        warnings.warn(
            "'generate_dataset_table' method not yet implemented for 'ImgSpiceArchive'"
        )
        return DataFrame()  # TEMPORARY


def load_archives() -> list[Archive]:
    archives: list[Archive] = []

    for planet in sd_planet.load_planets():
        for archive_path in planet.archives_path.iterdir():
            if not archive_path.is_dir():
                continue

            archive = Archive.create(archive_path, planet)
            archives.append(archive)

    return archives
