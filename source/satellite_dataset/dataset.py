from pathlib import Path
from typing import Iterator

import pandas as pd
from pandas import DataFrame, Series

import source.satellite_dataset.archive as sd_archive
from source.satellite_dataset.archive import Archive


class SatelliteDataset:
    def __init__(self, name: str, path: Path, archive: Archive) -> None:
        self.name = name
        self.archive = archive
        self.data_path = path / "data"

        self._table_path = path / "table.pkl"
        self._table = self._load_table()

    def __iter__(self) -> Iterator[Series]:
        for index, data in self._table.iterrows():
            yield data

    def __len__(self) -> int:
        return len(self._table)

    def generate_patches(
        self, scale_km: float, resolution: int, global_normalization: bool = False
    ) -> None:
        self.archive.generate_dataset_patches(
            self, scale_km, resolution, global_normalization=global_normalization
        )

    def generate_table(self) -> None:
        table = self.archive.generate_dataset_table(self)

        self._table_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_pickle(self._table_path)

    def validate(self) -> tuple[bool, str]:
        return self.archive.validate_dataset(self)

    def _load_table(self) -> DataFrame:
        if not self._table_path.is_file():
            self.generate_table()

        return pd.read_pickle(self._table_path)


def _build_dataset_registry() -> dict[str, SatelliteDataset]:
    print("Loading satellite datasets...")

    dataset_registry: dict[str, SatelliteDataset] = {}

    for archive in sd_archive.load_archives():
        for dataset_path in archive.datasets_path.iterdir():
            if not dataset_path.is_dir():
                continue

            dataset_name = dataset_path.name
            dataset = SatelliteDataset(dataset_name, dataset_path, archive)
            dataset_registry[dataset_name] = dataset

    return dataset_registry


_dataset_registry = _build_dataset_registry()


def get(name: str | None = None) -> SatelliteDataset:
    if name is None:
        print("\nAvailable datasets:\n-------------------")

        for dataset_name in _dataset_registry:
            print(dataset_name)

        print()

        name = input("Enter dataset name: ")

    return _dataset_registry[name]
