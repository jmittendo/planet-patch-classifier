from pathlib import Path

# Paths
OUTPUTS_DIR_PATH = Path(".")

# Patches
PATCH_RESOLUTION_TOLERANCE = 1.5
PATCH_BACKGROUND_ANGLE = 75
PATCH_OUTLIER_SIGMA = 3

# Misc
DOWNLOAD_CHUNK_SIZE = 67_108_864  # 64 MB
DATETIME_FORMAT = r"%Y-%m-%d_%H-%M-%S"
LOGGING_FORMAT = r"%(asctime)s - %(levelname)s - %(message)s"
JSON_INDENT = 4
