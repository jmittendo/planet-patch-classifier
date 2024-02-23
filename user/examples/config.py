from pathlib import Path

# Paths
DATA_DIR_PATH = Path(".")

# Patches
PATCH_RESOLUTION_TOLERANCE = 1.5
PATCH_ANGLE_THRESHOLD = 75
PATCH_OUTLIER_SIGMA = 5
MIN_PATCH_DENSITY = 0.5
NUM_PATCH_DENSITY_BINS = 8
MIN_PATCH_BIN_DENSITY = 0.1
PATCH_INTERPOLATION_METHOD = "nearest"

# Training
TRAIN_BATCH_SIZE = 32
TRAIN_EPOCHS = 128
TRAIN_OUTPUT_INTERVAL = 4

# Misc
DOWNLOAD_CHUNK_SIZE = 67_108_864  # 64 MB
DATETIME_FORMAT = r"%Y-%m-%d_%H-%M-%S"
LOGGING_FORMAT = r"%(asctime)s - %(levelname)s - %(message)s"
JSON_INDENT = 4
ENABLE_TEX_PLOTS = True
PLOT_DPI = 150
