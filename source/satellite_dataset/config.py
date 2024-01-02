from pathlib import Path

import user.config as ucfg

# Paths
DATASET_TABLES_DIR_PATH = ucfg.OUTPUTS_DIR_PATH / "satellite-dataset-tables"
DATASET_DOWNLOADS_DIR_PATH = ucfg.OUTPUTS_DIR_PATH / "satellite-datasets"
DATASET_DOWNLOADS_JSON_PATH = Path("resources/satellite-dataset-downloads.json")
DATASETS_JSON_PATH = Path("user/satellite-datasets.json")

# Constants
VENUS_RADIUS_M = 6.0518e6
