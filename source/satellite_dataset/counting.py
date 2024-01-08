import pandas as pd
from pandas import DataFrame

import source.satellite_dataset.config as sdcfg
import source.satellite_dataset.table as sd_table
import source.satellite_dataset.utility as sd_util
from source.satellite_dataset.typing import SatelliteDataset

DATASET_NAME = "vex-vmc-uv"


def count_dataset_files(dataset: SatelliteDataset, regenerate_table: bool) -> None:
    dataset_archive = sd_util.load_archive(dataset["archive"])
    table_path = sdcfg.DATASET_TABLES_DIR_PATH / f"{dataset['name']}.pkl"

    if regenerate_table or not table_path.is_file():
        sd_table.generate_dataset_table(dataset, dataset_archive, table_path)

    dataset_table: DataFrame = pd.read_pickle(table_path)

    print(f"Number of files in the dataset: {len(dataset_table)}")
