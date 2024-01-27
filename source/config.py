from pathlib import Path

import user.config as user_config

RESOURCES_DIR_PATH = Path("resources")
LOGS_DIR_PATH = user_config.DATA_DIR_PATH / "logs"
DATA_DIR_PATH = user_config.DATA_DIR_PATH / "data"
PLOTS_DIR_PATH = user_config.DATA_DIR_PATH / "plots"
