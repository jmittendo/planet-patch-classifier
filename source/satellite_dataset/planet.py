import json
from enum import StrEnum
from typing import TypedDict

import source.satellite_dataset.config as sd_config


class _PlanetDict(TypedDict):
    name: str
    radius_km: float
    rotation: str


class Planet:
    class Rotation(StrEnum):
        PROGRADE = "prograde"
        RETROGRADE = "retrograde"

    def __init__(
        self, name: str, radius_km: float, rotation: "Planet.Rotation"
    ) -> None:
        self.name = name
        self.radius_km = radius_km
        self.rotation = rotation

    @classmethod
    def from_dict(cls, planet_dict: _PlanetDict) -> "Planet":
        name = planet_dict["name"]
        radius_km = planet_dict["radius_km"]
        rotation = Planet.Rotation(planet_dict["rotation"])

        return cls(name, radius_km, rotation)


def load(name: str) -> Planet:
    with open(sd_config.PLANETS_JSON_PATH) as planets_json:
        planets_dict: dict[str, _PlanetDict] = json.load(planets_json)

    return Planet.from_dict(planets_dict[name])
