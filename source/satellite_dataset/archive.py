import json
from enum import StrEnum
from pathlib import Path
from typing import TypedDict

import source.satellite_dataset.config as sd_config
import source.satellite_dataset.planet as sd_planet
from source.satellite_dataset.planet import Planet


class ArchiveType(StrEnum):
    IMG_GEO = "img-geo"
    IMG_SPICE = "img-spice"


class _ArchiveDict(TypedDict):
    name: str
    type: str
    planet: str
    spice_path: str | None


class Archive:
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


def load(name: str) -> Archive:
    with open(sd_config.ARCHIVES_JSON_PATH) as archives_json:
        archives_dict: dict[str, _ArchiveDict] = json.load(archives_json)

    return Archive.from_dict(archives_dict[name])
