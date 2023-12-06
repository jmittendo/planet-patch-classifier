import json
import warnings
from pathlib import Path
from typing import TypedDict

import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

from source.utility import config

OUTPUT_DIR = "datasets/satellite-datasets/vex-test"
DATASET_NAME = "vex-vmc-uv"


ArchiveInfo = TypedDict("ArchiveInfo", {"url": str, "href-filter": str})
DatasetInfo = TypedDict("DatasetInfo", {"archive": str, "file-name-filter": str})


class DownloadsConfig(TypedDict):
    archives: dict[str, ArchiveInfo]
    datasets: dict[str, DatasetInfo]


def main():
    with open(config.download_configs_json_path, "r") as json_file:
        downloads_config: DownloadsConfig = json.load(json_file)

    dataset_info = downloads_config["datasets"][DATASET_NAME]
    archive_name = dataset_info["archive"]
    file_name_filter = dataset_info["file-name-filter"]
    archive_info = downloads_config["archives"][archive_name]
    archive_url = archive_info["url"]
    href_filter = archive_info["href-filter"]

    match archive_name:
        case "vex-vmc":
            download_vex_vmc_dataset(archive_url, href_filter, file_name_filter)
        case _:
            raise ValueError(
                f"No download script implemented for archive with name '{archive_name}'"
            )


def download_vex_vmc_dataset(
    archive_url: str, href_filter: str, file_name_filter: str
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
        mission_dir_path = Path(OUTPUT_DIR) / mission_dir_name

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
