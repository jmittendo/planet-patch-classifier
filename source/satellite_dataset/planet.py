# This file is part of planet-patch-classifier, a Python tool for generating and
# classifying planet patches from satellite imagery via unsupervised machine learning
# Copyright (C) 2024  Jan Mittendorf (jmittendo)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import json
from enum import StrEnum
from pathlib import Path
from typing import TypedDict

import source.satellite_dataset.config as sd_config


class _PlanetInfoDict(TypedDict):
    radius_km: float
    rotation: str


class Planet:
    class Rotation(StrEnum):
        PROGRADE = "prograde"
        RETROGRADE = "retrograde"

    def __init__(self, dir_path: Path) -> None:
        self.name = dir_path.name
        self.archives_path = dir_path / "archives"

        planet_info_json_path = dir_path / "planet.json"

        with open(planet_info_json_path) as planet_info_json:
            planet_info: _PlanetInfoDict = json.load(planet_info_json)

        self.radius_km = planet_info["radius_km"]
        self.rotation = Planet.Rotation(planet_info["rotation"])


def load_planets() -> list[Planet]:
    planets: list[Planet] = []

    for planet_dir_path in sd_config.DATASETS_DIR_PATH.iterdir():
        if not planet_dir_path.is_dir():
            continue

        planet = Planet(planet_dir_path)
        planets.append(planet)

    return planets
