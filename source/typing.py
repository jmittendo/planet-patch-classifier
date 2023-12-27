from typing import TypedDict


class DownloadConfig(TypedDict):
    archive: str
    instrument: str
    wavelengths: list[str]


class SatelliteDataset(TypedDict):
    path: str
    archive: str
