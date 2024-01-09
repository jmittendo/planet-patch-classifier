import abc
import json
import typing
from abc import ABC
from enum import StrEnum
from pathlib import Path
from typing import TypedDict

from pandas import DataFrame

import source.satellite_dataset.config as sd_config
import source.satellite_dataset.planet as sd_planet
import source.satellite_dataset.table as sd_table
import source.satellite_dataset.validation as sd_validation
from source.satellite_dataset.dataset import Dataset
from source.satellite_dataset.planet import Planet


class _ArchiveDict(TypedDict):
    name: str
    type: str
    planet: str
    spice_path: str | None


class Archive(ABC):
    class Type(StrEnum):
        IMG_GEO = "img-geo"
        IMG_SPICE = "img-spice"

    def __init__(
        self, name: str, type: "Archive.Type", planet: Planet, spice_path: Path | None
    ) -> None:
        self.name = name
        self.type = type
        self.planet = planet
        self.spice_path = spice_path

    @classmethod
    def from_dict(cls, archive_dict: _ArchiveDict) -> "Archive":
        name = archive_dict["name"]
        type = Archive.Type[archive_dict["type"]]
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
        if self.spice_path is None:
            raise ValueError(f"Spice path must not be 'null' for archive {self.name}")

        return sd_table.generate_vex_vmc_dataset_table(dataset, self.spice_path)


class VcoArchive(Archive):
    @typing.override
    def validate_dataset(self, dataset: Dataset) -> tuple[bool, str]:
        return sd_validation.validate_vco_dataset(dataset)

    @typing.override
    def generate_dataset_table(self, dataset: Dataset) -> DataFrame:
        return sd_table.generate_vco_dataset_table(dataset, self.planet.radius_km)


def load(name: str) -> Archive:
    with open(sd_config.ARCHIVES_JSON_PATH) as archives_json:
        archives_dict: dict[str, _ArchiveDict] = json.load(archives_json)

    return Archive.from_dict(archives_dict[name])
