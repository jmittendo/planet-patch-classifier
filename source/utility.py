from configparser import ConfigParser
from pathlib import Path
from typing import TypedDict

CONFIG_PATH = "config.cfg"


class SatelliteDataset(TypedDict):
    path: str
    satellite: str


class Config:
    def __init__(self, config_path: str) -> None:
        cfg_parser = ConfigParser()
        cfg_parser.read(config_path)

        self.downloads_dir = Path(cfg_parser["Paths"]["downloads_dir"])
        self.download_configs_json_path = Path(
            cfg_parser["Paths"]["download_configs_json"]
        )
        self.satellite_datasets_json_path = Path(
            cfg_parser["Paths"]["satellite_datasets_json"]
        )
        self.json_indent = cfg_parser["Misc"].getint("json_indent")


config = Config(CONFIG_PATH)


def user_confirm(message: str) -> bool:
    reply = input(f"{message} (y/n): ")

    while reply.lower() not in ("y", "n", "yes", "no"):
        reply = input("Please reply with 'yes'/'y' or 'no'/'n': ")

    return reply.lower() in ("y", "yes")
