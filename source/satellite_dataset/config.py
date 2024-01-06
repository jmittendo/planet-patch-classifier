import source.config as cfg

# Paths
DATASET_TABLES_DIR_PATH = cfg.DATA_DIR_PATH / "satellite-dataset-tables"
DATASET_DOWNLOADS_DIR_PATH = cfg.DATA_DIR_PATH / "satellite-datasets"
DATASET_DOWNLOADS_JSON_PATH = (
    cfg.RESOURCES_DIR_PATH / "satellite-dataset-downloads.json"
)
DATASETS_JSON_PATH = cfg.USER_DIR_PATH / "satellite-datasets.json"
ARCHIVES_JSON_PATH = cfg.USER_DIR_PATH / "satellite-data-archives.json"
PATCHES_DIR_PATH = cfg.DATA_DIR_PATH / "patch-datasets"
