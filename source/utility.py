import logging
from datetime import datetime
from pathlib import Path

import source.config as config
import user.config as user_config


def user_confirm(message: str) -> bool:
    reply = input(f"{message} (y/n): ")

    while reply.lower() not in ("y", "n", "yes", "no"):
        reply = input("Please reply with 'yes'/'y' or 'no'/'n': ")

    return reply.lower() in ("y", "yes")


def configure_logging(log_file_name_base: str) -> None:
    current_datetime_str = datetime.now().strftime(user_config.DATETIME_FORMAT)

    logs_dir_path = config.LOGS_DIR_PATH
    logs_dir_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=Path(
            logs_dir_path, f"{log_file_name_base}_{current_datetime_str}.log"
        ),
        format=user_config.LOGGING_FORMAT,
        level=logging.DEBUG,
    )
