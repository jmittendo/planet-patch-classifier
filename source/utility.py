import logging
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path
from typing import TypedDict

CONFIG_PATH = "config.cfg"


class SatelliteDataset(TypedDict):
    path: str
    satellite: str


class Config:
    def __init__(self, config_path: str) -> None:
        cfg_parser = ConfigParser(inline_comment_prefixes="#", interpolation=None)
        cfg_parser.read(config_path)

        self.logs_dir_path = Path(cfg_parser["Paths"]["logs_dir"])
        self.downloads_dir_path = Path(cfg_parser["Paths"]["downloads_dir"])
        self.download_configs_json_path = Path(
            cfg_parser["Paths"]["download_configs_json"]
        )
        self.satellite_datasets_json_path = Path(
            cfg_parser["Paths"]["satellite_datasets_json"]
        )
        self.download_chunk_size = cfg_parser["Misc"].getint("download_chunk_size")
        self.datetime_format = cfg_parser["Misc"]["datetime_format"]
        self.logging_format = cfg_parser["Misc"]["logging_format"]
        self.json_indent = cfg_parser["Misc"].getint("json_indent")


config = Config(CONFIG_PATH)


def user_confirm(message: str) -> bool:
    reply = input(f"{message} (y/n): ")

    while reply.lower() not in ("y", "n", "yes", "no"):
        reply = input("Please reply with 'yes'/'y' or 'no'/'n': ")

    return reply.lower() in ("y", "yes")


def configure_logging(log_file_name_base: str) -> None:
    current_datetime_str = datetime.now().strftime(config.datetime_format)

    logging.basicConfig(
        filename=Path(
            config.logs_dir_path, f"{log_file_name_base}_{current_datetime_str}.log"
        ),
        format=config.logging_format,
        level=logging.DEBUG,
    )
