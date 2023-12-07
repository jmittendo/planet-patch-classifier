import json
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import TypedDict

import requests
from bs4 import BeautifulSoup, Tag
from requests import HTTPError
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


def main() -> None:
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
            href_filter = filter_patterns[0]

            download_vex_vmc_dataset(
                archive_url,
                href_filter,
                dataset_identifier,
                output_dir_path,
                config.download_chunk_size,
            )
        case "vco":
            img_href_filter, geo_href_filter = [
                fp.replace("[identifier]", dataset_identifier) for fp in filter_patterns
            ]

            download_vco_dataset(
                archive_url,
                img_href_filter,
                geo_href_filter,
                output_dir_path,
                config.download_chunk_size,
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
    archive_url: str,
    href_filter: str,
    file_name_filter: str,
    output_dir_path: Path,
    chunk_size: int,
) -> None:
    archive_url_stripped = archive_url.rstrip("/")

    mission_dir_names = [
        href.rstrip("/")
        for href in get_hrefs(archive_url, dir_only=True)
        if href_filter in href
    ]

    for mission_dir_name in tqdm(
        mission_dir_names, desc="Full download progress", leave=False
    ):
        mission_dir_url = f"{archive_url_stripped}/{mission_dir_name}"
        mission_dir_path = output_dir_path / mission_dir_name

        img_dir_url = f"{mission_dir_url}/DATA"

        orbit_dir_names = [
            href.rstrip("/")
            for href in get_hrefs(img_dir_url, dir_only=True)
            if mission_dir_name not in href
        ]

        for orbit_dir_name in tqdm(
            orbit_dir_names,
            desc=f"└ Mission extension directory '{mission_dir_name}' progress",
            leave=False,
        ):
            orbit_dir_url = f"{img_dir_url}/{orbit_dir_name}"

            img_dir_path = mission_dir_path / "DATA" / orbit_dir_name
            img_dir_path.mkdir(parents=True, exist_ok=True)

            geo_dir_path = mission_dir_path / "GEOMETRY" / orbit_dir_name
            geo_dir_path.mkdir(parents=True, exist_ok=True)

            img_file_names = [
                href
                for href in get_hrefs(orbit_dir_url)
                if file_name_filter in href and ".IMG" in href
            ]

            for img_file_name in tqdm(
                img_file_names,
                desc=f"  └ Orbit directory '{orbit_dir_name}' progress",
                leave=False,
            ):
                img_file_url = f"{orbit_dir_url}/{img_file_name}"
                img_file_path = img_dir_path / img_file_name

                geo_file_name = img_file_name.rstrip(".IMG") + ".GEO"
                geo_file_url = (
                    f"{mission_dir_url}/GEOMETRY/{orbit_dir_name}/{geo_file_name}"
                )
                geo_file_path = geo_dir_path / geo_file_name

                # If geometry file (url) does not exist skip both downloads
                # Note: Considering the code above, this can happen when an image file
                # does not have a corresponding geometry file and is therefore not a
                # critical error.
                try:
                    download_file(
                        geo_file_url,
                        geo_file_path,
                        chunk_size=chunk_size,
                        pbar_indent=2,
                    )
                except HTTPError:
                    continue

                # If image file (url) does not exist skip download
                # Note: Unlike in the case of the geometry files this should never
                # happen as per the code above. However, if for some unforeseen reason
                # it happens anyway, we do not want to stop the whole download process
                # and simply warn the user.
                try:
                    download_file(
                        img_file_url,
                        img_file_path,
                        chunk_size=chunk_size,
                        pbar_indent=2,
                    )
                except HTTPError:
                    warnings.warn(
                        "Error while trying to download image file at url "
                        f"'{img_file_url}'"
                    )


def download_vco_dataset(
    archive_url: str,
    img_href_filter: str,
    geo_href_filter: str,
    output_dir_path: Path,
    chunk_size: int,
) -> None:
    def download_dirs(
        href_filter: str, file_type: str, sub_dir: str | None = None
    ) -> None:
        archive_url_stripped = archive_url.rstrip("/")
        url = f"{archive_url_stripped}/{sub_dir or ''}"
        url_stripped = url.rstrip("/")

        dir_names = [
            href.rstrip("/")
            for href in get_hrefs(url, dir_only=True)
            if href_filter in href
        ]

        for dir_name in tqdm(
            dir_names,
            desc=f"{file_type.capitalize()} files download progress",
            leave=False,
        ):
            dir_url = f"{url_stripped}/{dir_name}"

            dir_path = output_dir_path / (sub_dir or "") / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)

            zip_file_names = [
                href
                for href in get_hrefs(dir_url)
                if (".zip" in href or ".tar.gz" in href or ".tar.xz" in href)
                and "netcdf" not in href
            ]

            for zip_file_name in tqdm(
                zip_file_names,
                desc=f"└ {file_type.capitalize()} file directory '{dir_name}' progress",
                leave=False,
            ):
                zip_file_url = f"{dir_url}/{zip_file_name}"
                zip_file_path = dir_path / zip_file_name

                # If zip file (url) does not exist skip download
                # Note: This should never happen as per the code above. However, if for
                # some unforeseen reason it happens anyway, we do not want to stop the
                # whole download process and simply warn the user.
                try:
                    download_file(
                        zip_file_url,
                        zip_file_path,
                        chunk_size=chunk_size,
                        pbar_indent=1,
                    )
                except HTTPError:
                    warnings.warn(
                        f"Error while trying to download {file_type.lower()} zip file "
                        f"at url '{zip_file_url}'"
                    )

    download_dirs(img_href_filter, "image")
    download_dirs(geo_href_filter, "geometry", sub_dir="extras")


def get_hrefs(url: str, dir_only: bool = False) -> list[str]:
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

        hrefs.append(href)

    return hrefs


def download_file(file_url: str, file_path: Path, *, chunk_size: int, pbar_indent: int):
    with requests.get(file_url, stream=True) as file_response:
        file_response.raise_for_status()
        file_size = file_response.headers.get("content-length")

        progress_bar = (
            None
            if file_size is None
            else tqdm(
                total=int(file_size),
                desc=f"{'  ' * pbar_indent}└ File '{file_path.name}' progress",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            )
        )

        with open(file_path, "wb") as file:
            for chunk in file_response.iter_content(chunk_size=chunk_size):
                file.write(chunk)

                if progress_bar is not None:
                    progress_bar.update(chunk_size)


if __name__ == "__main__":
    main()
