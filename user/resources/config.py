from pathlib import Path

# PATHS --------------------------------------------------------------------------------
DATA_DIR_PATH = Path(".")

# PATCHES ------------------------------------------------------------------------------
# Only satellite images with a maximum spatial resolution lower (i.e. better) than
# patch scale / patch resolution * PATCH_RESOLUTION_TOLERANCE will be considered
PATCH_RESOLUTION_TOLERANCE = 1.5

PATCH_ANGLE_THRESHOLD = 75
PATCH_OUTLIER_SIGMA = 5
MIN_PATCH_DENSITY = 0.5
NUM_PATCH_DENSITY_BINS = 8
MIN_PATCH_BIN_DENSITY = 0.1
PATCH_INTERPOLATION_METHOD = "nearest"
SPICE_ABBERATION_CORRECTION = "LT+S"

# TRAINING -----------------------------------------------------------------------------
TRAIN_BATCH_SIZE = 32
TRAIN_EPOCHS = 64
TRAIN_OUTPUT_INTERVAL = 4
TRAIN_TEST_RATIO = 0.2
SIMCLR_LOSS_TEMPERATURE = 1

# BENCHMARK ----------------------------------------------------------------------------
BENCHMARK_ITERATIONS = 8
BENCHMARK_DATASET_NAME = "cloud-imvn-1.0"
BENCHMARK_DATASET_VERSION = "default"
BENCHMARK_ENCODER_MODELS = ["simple", "autoencoder", "simclr"]
BENCHMARK_ENCODER_BASE_MODEL = "resnet18"
BENCHMARK_REDUCTION_METHODS = ["tsne", "pca", None]
BENCHMARK_CLUSTERING_METHODS = ["kmeans", "hac"]
BENCHMARK_PCA_DIM_VALUES = [256, 64, 16]

# PLOTS --------------------------------------------------------------------------------
PLOT_FONT = "Times New Roman"
PLOT_MATH_FONT = "stix"
PLOT_FONT_SIZE = 8
PLOT_DPI = 1000
PLOT_ENABLE_TEX = False

# MISC ---------------------------------------------------------------------------------
JSON_INDENT = 4
