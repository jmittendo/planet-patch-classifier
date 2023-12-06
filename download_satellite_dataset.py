import json
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import TypedDict

import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

import source.utility as util
from source.utility import config

ArchiveInfo = TypedDict("ArchiveInfo", {"url": str, "filter-patterns": list[str]})


class DatasetInfo(TypedDict):
    archive: str
    identifier: str


class DownloadConfigs(TypedDict):
    archives: dict[str, ArchiveInfo]
    datasets: dict[str, DatasetInfo]


def main():
    with open(config.download_configs_json_path, "r") as json_file:
        download_configs: DownloadConfigs = json.load(json_file)

    input_args = parse_input_args()
    dataset_name: str | None = input_args.name

    if dataset_name is None:
        if util.user_confirm("Display available datasets?"):
            print("\nAvailable datasets:\n-------------------")

            for dataset_name in download_configs["datasets"]:
                print(dataset_name)

            print()

        dataset_name = input("Enter dataset name: ")

    dataset_info = download_configs["datasets"][dataset_name]
    archive_name = dataset_info["archive"]
    dataset_identifier = dataset_info["identifier"]
    archive_info = download_configs["archives"][archive_name]
    archive_url = archive_info["url"]
    filter_patterns = archive_info["filter-patterns"]

    output_dir_path = config.downloads_dir / dataset_name

    match archive_name:
        case "vex-vmc":
            download_vex_vmc_dataset(
                archive_url, filter_patterns[0], dataset_identifier, output_dir_path
            )
        case _:
            raise ValueError(
                f"No download script implemented for archive with name '{archive_name}'"
            )


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description="Download a named dataset. Run without arguments to get a list of "
        "available datasets",
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")

    return arg_parser.parse_args()


def download_vex_vmc_dataset(
    archive_url: str, href_filter: str, file_name_filter: str, output_dir_path: Path
) -> None:
    url_stripped = archive_url.rstrip("/")

    mission_dir_names = [
        href.rstrip("/")
        for href in get_hrefs(
            archive_url, dir_only=True, positive_filters=[href_filter]
        )
    ]

    for mission_dir_name in tqdm(
        mission_dir_names, desc="Full download progress", leave=False
    ):
        mission_dir_url = f"{url_stripped}/{mission_dir_name}"
        mission_dir_path = output_dir_path / mission_dir_name

        img_dir_url = f"{mission_dir_url}/DATA"

        orbit_dir_names = [
            href.rstrip("/")
            for href in get_hrefs(
                img_dir_url, dir_only=True, negative_filters=[mission_dir_name]
            )
        ]

        for orbit_dir_name in tqdm(
            orbit_dir_names,
            desc=f"└ Mission extension directory '{mission_dir_name}' progress",
            leave=False,
        ):
            orbit_dir_url = f"{img_dir_url}/{orbit_dir_name}"

            img_file_dir_path = mission_dir_path / "DATA" / orbit_dir_name
            img_file_dir_path.mkdir(parents=True, exist_ok=True)

            geo_file_dir_path = mission_dir_path / "GEOMETRY" / orbit_dir_name
            geo_file_dir_path.mkdir(parents=True, exist_ok=True)

            img_file_names = get_hrefs(
                orbit_dir_url, positive_filters=[file_name_filter, ".IMG"]
            )

            for img_file_name in tqdm(
                img_file_names,
                desc=f"  └ Orbit directory '{orbit_dir_name}' progress",
                leave=False,
            ):
                img_file_url = f"{orbit_dir_url}/{img_file_name}"

                geo_file_name = img_file_name.rstrip(".IMG") + ".GEO"
                geo_file_url = (
                    f"{mission_dir_url}/GEOMETRY/{orbit_dir_name}/{geo_file_name}"
                )

                geo_file_response = requests.get(geo_file_url)

                # If geometry file (url) does not exist skip downloads
                if geo_file_response.status_code >= 400:
                    continue

                img_file_response = requests.get(img_file_url)

                # If image file (url) does not exist skip downloads
                if img_file_response.status_code >= 400:
                    continue

                img_file_path = img_file_dir_path / img_file_name
                geo_file_path = geo_file_dir_path / geo_file_name

                with open(img_file_path, "wb") as img_file:
                    img_file.write(img_file_response.content)

                with open(geo_file_path, "wb") as geo_file:
                    geo_file.write(geo_file_response.content)


def download_vco_dataset():
    ...


def get_hrefs(
    url: str,
    dir_only: bool = False,
    positive_filters: list[str] | None = None,
    negative_filters: list[str] | None = None,
) -> list[str]:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    hrefs: list[str] = []

    a_tag: Tag
    for a_tag in soup.find_all("a"):
        href = a_tag.get("href")

        if not isinstance(href, str):
            warnings.warn("'href' attribute of html 'a' tag not of type 'str'")
            continue

        if dir_only and href[-1] != "/":
            continue

        if positive_filters is not None and not all(
            pf in href for pf in positive_filters
        ):
            continue

        if negative_filters is not None and any(nf in href for nf in negative_filters):
            continue

        hrefs.append(href)

    return hrefs


if __name__ == "__main__":
    main()
