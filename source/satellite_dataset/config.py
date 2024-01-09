import source.config as config

# Paths
DATASET_TABLES_DIR_PATH = config.DATA_DIR_PATH / "satellite-dataset-tables"
DATASET_DOWNLOADS_DIR_PATH = config.DATA_DIR_PATH / "satellite-datasets"
DATASET_DOWNLOADS_JSON_PATH = (
    config.RESOURCES_DIR_PATH / "satellite-dataset-downloads.json"
)
DATASETS_JSON_PATH = config.USER_DIR_PATH / "satellite-datasets.json"
ARCHIVES_JSON_PATH = config.USER_DIR_PATH / "satellite-data-archives.json"
PLANETS_JSON_PATH = config.RESOURCES_DIR_PATH / "planets.json"
