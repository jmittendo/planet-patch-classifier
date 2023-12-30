from pathlib import Path

# Paths
LOGS_DIR_PATH = Path("logs")
DOWNLOADS_DIR_PATH = Path("datasets/satellite-datasets")
SATELLITE_DATASET_TABLES_DIR_PATH = Path("datasets/satellite-datasets/tables")
DOWNLOAD_CONFIGS_JSON_PATH = Path("datasets/download-configs.json")
SATELLITE_DATASETS_JSON_PATH = Path("user/satellite-datasets.json")

# Patches
PATCH_RESOLUTION_TOLERANCE = 1.5

# Misc
DOWNLOAD_CHUNK_SIZE = 67_108_864  # 64 MB
DATETIME_FORMAT = r"%Y-%m-%d_%H-%M-%S"
LOGGING_FORMAT = r"%(asctime)s - %(levelname)s - %(message)s"
JSON_INDENT = 4
