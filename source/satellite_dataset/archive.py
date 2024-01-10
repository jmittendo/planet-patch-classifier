import abc
import json
import typing
from abc import ABC
from pathlib import Path
from typing import TypedDict

from pandas import DataFrame

import source.satellite_dataset.config as sd_config
import source.satellite_dataset.patches as sd_patches
import source.satellite_dataset.planet as sd_planet
import source.satellite_dataset.table as sd_table
import source.satellite_dataset.validation as sd_validation
from source.patch_dataset.typing import PatchNormalization
from source.satellite_dataset.planet import Planet
from source.satellite_dataset.typing import ImgGeoDataArrays

if typing.TYPE_CHECKING:
    from source.satellite_dataset.dataset import Dataset


class _ArchiveDict(TypedDict):
    name: str
    planet: str
    spice_path: str | None


class Archive(ABC):
    _subclass_registry: dict[str, type["Archive"]] = {}

    def __init__(self, name: str, planet: Planet, spice_path: Path | None) -> None:
        self.name = name
        self.planet = planet
        self.spice_path = spice_path

    @abc.abstractmethod
    def __init_subclass__(cls, name: str) -> None:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, archive_dict: _ArchiveDict) -> "Archive":
        name = archive_dict["name"]
        planet = sd_planet.load(archive_dict["planet"])

        spice_path_str = archive_dict["spice_path"]
        spice_path = None if spice_path_str is None else Path(spice_path_str)

        archive_subclass = cls._subclass_registry[name]

        return archive_subclass(name, planet, spice_path)

    @abc.abstractmethod
    def validate_dataset(self, dataset: "Dataset") -> tuple[bool, str]:
        raise NotImplementedError

    @abc.abstractmethod
    def generate_dataset_table(self, dataset: "Dataset") -> DataFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def generate_dataset_patches(
        self,
        dataset: "Dataset",
        patch_scale_km: float,
        patch_resolution: int,
        patch_normalization: PatchNormalization,
    ) -> None:
        raise NotImplementedError


class ImgGeoArchive(Archive):
    @typing.override
    def __init_subclass__(cls, name: str) -> None:
        cls._subclass_registry[name] = cls

    @typing.override
    def generate_dataset_patches(
        self,
        dataset: "Dataset",
        patch_scale_km: float,
        patch_resolution: int,
        patch_normalization: PatchNormalization,
    ) -> None:
        sd_patches.generate_img_geo_patches(
            self, dataset, patch_scale_km, patch_resolution, patch_normalization
        )

    @abc.abstractmethod
    def load_data_arrays(
        self, img_file_path: Path, geo_file_path: Path
    ) -> ImgGeoDataArrays:
        raise NotImplementedError


class ImgSpiceArchive(Archive):
    @typing.override
    def __init_subclass__(cls, name: str) -> None:
        cls._subclass_registry[name] = cls

    @typing.override
    def generate_dataset_patches(
        self,
        dataset: "Dataset",
        patch_scale_km: float,
        patch_resolution: int,
        patch_normalization: PatchNormalization,
    ) -> None:
        ...


class VexVmcArchive(ImgGeoArchive, name="vex-vmc"):
    @typing.override
    def validate_dataset(self, dataset: "Dataset") -> tuple[bool, str]:
        return sd_validation.validate_vex_vmc_dataset(dataset)

    @typing.override
    def generate_dataset_table(self, dataset: "Dataset") -> DataFrame:
        if self.spice_path is None:
            raise ValueError(f"Spice path must not be 'null' for archive {self.name}")

        return sd_table.generate_vex_vmc_dataset_table(dataset, self.spice_path)

    @typing.override
    def load_data_arrays(
        self, img_file_path: Path, geo_file_path: Path
    ) -> ImgGeoDataArrays:
        return sd_patches.load_vex_vmx_data_arrays(img_file_path, geo_file_path)


class VcoArchive(ImgGeoArchive, name="vco"):
    @typing.override
    def validate_dataset(self, dataset: "Dataset") -> tuple[bool, str]:
        return sd_validation.validate_vco_dataset(dataset)

    @typing.override
    def generate_dataset_table(self, dataset: "Dataset") -> DataFrame:
        return sd_table.generate_vco_dataset_table(dataset, self.planet.radius_km)

    @typing.override
    def load_data_arrays(
        self, img_file_path: Path, geo_file_path: Path
    ) -> ImgGeoDataArrays:
        return sd_patches.load_vco_data_arrays(img_file_path, geo_file_path)


class JunoJncArchive(ImgSpiceArchive, name="juno-jnc"):
    @typing.override
    def validate_dataset(self, dataset: "Dataset") -> tuple[bool, str]:
        ...

    @typing.override
    def generate_dataset_table(self, dataset: "Dataset") -> DataFrame:
        ...


def _build_archive_registry() -> dict[str, Archive]:
    archive_registry: dict[str, Archive] = {}

    with open(sd_config.ARCHIVES_JSON_PATH) as archives_json:
        archive_dicts: list[_ArchiveDict] = json.load(archives_json)

        for archive_dict in archive_dicts:
            archive_name = archive_dict["name"]
            archive_registry[archive_name] = Archive.from_dict(archive_dict)

    return archive_registry


_archive_registry = _build_archive_registry()


def get(name: str) -> Archive:
    return _archive_registry[name]
