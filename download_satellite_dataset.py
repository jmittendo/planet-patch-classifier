# NOTE:
# I have attempted to write this script in such a way that as long as the archive
# structures or naming conventions don't change, it will find newly added files, even if
# they weren't present at the time of this file's creation.
#
# However: As per nature of a script that attempts automatically finding and downloading
# files from an online archive, the code contains many hardcoded URLs, relative paths
# and name patterns that are sensitive to change. It is therefore possible that this
# script will stop working at some point. In that case you may of course try fixing the
# code yourself, or resort to manually downloading the necessary files. The README.md
# file should contain instructions on how to structure/format the downloaded archives so
# that they can be used with the provided scripts.


import json
import shutil
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Literal, TypedDict

import requests
from bs4 import BeautifulSoup, Tag
from requests import HTTPError
from tqdm import tqdm

import source.utility as util
from source.utility import config


class DownloadConfig(TypedDict):
    archive: str
    instrument: str
    wavelengths: list[str]


type SatelliteFileType = Literal["image", "geometry"]


def main() -> None:
    with open(config.download_configs_json_path, "r") as json_file:
        download_configs: dict[str, DownloadConfig] = json.load(json_file)

    input_args = parse_input_args()
    dataset_name: str | None = input_args.name

    if dataset_name is None:
        if util.user_confirm("Display available datasets?"):
            print("\nAvailable datasets:\n-------------------")

            for dataset_name in download_configs:
                print(dataset_name)

            print()

        dataset_name = input("Enter dataset name: ")

    download_config = download_configs[dataset_name]
    dataset_archive = download_config["archive"]
    dataset_instrument = download_config["instrument"]
    dataset_wavelengths = download_config["wavelengths"]

    output_dir_path = config.downloads_dir_path / dataset_name

    match dataset_archive:
        case "vex-vmc":
            download_vex_vmc_dataset(
                dataset_wavelengths[0],
                output_dir_path,
                config.download_chunk_size,
            )
        case "vco":
            download_vco_dataset(
                dataset_instrument,
                dataset_wavelengths,
                dataset_name,
                config.download_chunk_size,
            )
        case _:
            raise ValueError(
                "No download script implemented for dataset archive "
                f"'{dataset_archive}'"
            )


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description="Download a named dataset. Run without arguments to get a list of "
        "available datasets",
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")

    return arg_parser.parse_args()


def download_vex_vmc_dataset(
    wavelength_filter: str, output_dir_path: Path, chunk_size: int
) -> None:
    archive_url = "https://archives.esac.esa.int/psa/ftp/VENUS-EXPRESS/VMC/"
    archive_url_stripped = archive_url.rstrip("/")

    href_filter = "VEX-V-VMC-3-RDR"

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
                if wavelength_filter in href and ".IMG" in href
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
                # NOTE: Considering the code above, this can happen when an image file
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
                # NOTE: Unlike in the case of the geometry files this should never
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
    instrument: str,
    wavelength_filters: list[str],
    dataset_name: str,
    chunk_size: int,
) -> None:
    archive_url = "https://data.darts.isas.jaxa.jp/pub/pds3/"
    archive_url_stripped = archive_url.rstrip("/")

    temp_output_dir_path = config.downloads_dir_path / dataset_name

    wavelength_output_dir_paths = {
        wavelength_filter: (
            config.downloads_dir_path / f"{dataset_name}-{wavelength_filter}"
        )
        for wavelength_filter in wavelength_filters
    }

    img_href_filter = f"vco-v-{instrument}-3-cdr-v"

    img_file_dir_names = [
        href.rstrip("/")
        for href in get_hrefs(archive_url_stripped, dir_only=True)
        if img_href_filter in href
    ]

    for img_file_dir_name in tqdm(
        img_file_dir_names, desc="Image files download progress", leave=False
    ):
        img_file_dir_version_str = img_file_dir_name.split("-")[5]

        img_file_dir_url = f"{archive_url_stripped}/{img_file_dir_name}"

        img_file_dir_path = temp_output_dir_path / img_file_dir_name
        img_file_dir_path.mkdir(parents=True, exist_ok=True)

        geo_file_dir_name = f"vco_{instrument}_l3_{img_file_dir_version_str}"
        geo_file_dir_url = f"{archive_url_stripped}/extras/{geo_file_dir_name}"

        geo_file_dir_path = temp_output_dir_path / "extras" / geo_file_dir_name
        geo_file_dir_path.mkdir(parents=True, exist_ok=True)

        img_zip_file_names = [
            href
            for href in get_hrefs(img_file_dir_url)
            if (".zip" in href or ".tar.gz" in href or ".tar.xz" in href)
            and "netcdf" not in href
        ]

        for img_zip_file_name in tqdm(
            img_zip_file_names,
            desc=f"└ Image file directory '{img_file_dir_name}' progress",
            leave=False,
        ):
            img_zip_file_url = f"{img_file_dir_url}/{img_zip_file_name}"
            img_zip_file_path = img_file_dir_path / img_zip_file_name

            geo_zip_file_name_stem = (
                f"{img_zip_file_name.split('.')[0].replace('_1', '_7')}_l3x_fits"
            )

            # If geometry zip file (url) does not exist skip both downloads
            # NOTE: Considering the code above, this can happen when an image zip file
            # does not have a corresponding geometry zip file and is therefore not a
            # critical error.
            geo_zip_file_download_successful = False

            for geo_zip_file_suffix in [".zip", ".tar.gz", ".tar.xz"]:
                geo_zip_file_name = geo_zip_file_name_stem + geo_zip_file_suffix
                geo_zip_file_url = f"{geo_file_dir_url}/{geo_zip_file_name}"
                geo_zip_file_path = geo_file_dir_path / geo_zip_file_name

                try:
                    download_file(
                        geo_zip_file_url,
                        geo_zip_file_path,
                        chunk_size=chunk_size,
                        pbar_indent=1,
                    )
                    geo_zip_file_download_successful = True
                    break
                except HTTPError:
                    print("Test")
                    continue

            if not geo_zip_file_download_successful:
                continue

            # If image zip file (url) does not exist skip download
            # NOTE: Unlike in the case of the geometry zip files this should never
            # happen as per the code above. However, if for some unforeseen reason
            # it happens anyway, we do not want to stop the whole download process
            # and simply warn the user.
            try:
                download_file(
                    img_zip_file_url,
                    img_zip_file_path,
                    chunk_size=chunk_size,
                    pbar_indent=1,
                )
            except HTTPError:
                warnings.warn(
                    f"Error while trying to download image zip file at url"
                    f"'{img_zip_file_url}'"
                )
                continue

            # Unpack the zip files with automatic format detection and delete them after
            shutil.unpack_archive(img_zip_file_path, img_file_dir_path)
            img_zip_file_path.unlink(missing_ok=True)

            shutil.unpack_archive(geo_zip_file_path, geo_file_dir_path)
            geo_zip_file_path.unlink(missing_ok=True)

            unpacked_img_file_dir_name = "_".join(img_zip_file_path.stem.split("_")[:2])
            unpacked_geo_file_dir_name = "_".join(geo_zip_file_path.stem.split("_")[:2])

            img_file_orbit_dirs_path = (
                img_file_dir_path / unpacked_img_file_dir_name / "data" / "l2b"
            )
            geo_file_orbit_dirs_path = (
                geo_file_dir_path
                / unpacked_geo_file_dir_name
                / "data"
                / "l3bx"
                / "fits"
            )

            for img_file_orbit_dir_path in img_file_orbit_dirs_path.iterdir():
                if not img_file_orbit_dir_path.is_dir():
                    continue

                orbit_dir_name = img_file_orbit_dir_path.name

                for wavelength_filter in wavelength_filters:
                    wavelength_output_dir_path = wavelength_output_dir_paths[
                        wavelength_filter
                    ]

                    # Find only the highest version for each file and save its path
                    img_file_highest_version_dict: dict[str, tuple[int, Path]] = {}

                    for img_file_path in img_file_orbit_dir_path.glob(
                        f"*{wavelength_filter}*.fit"
                    ):
                        img_file_stem_components = img_file_path.stem.split("_")
                        img_file_name_base = "_".join(img_file_stem_components[:5])
                        img_file_version = int(img_file_stem_components[5].lstrip("v"))

                        current_highest_version_tuple = (
                            img_file_highest_version_dict.get(img_file_name_base)
                        )
                        current_highest_version = (
                            current_highest_version_tuple[0]
                            if current_highest_version_tuple is not None
                            else 0
                        )

                        if img_file_version > current_highest_version:
                            img_file_highest_version_dict[img_file_name_base] = (
                                img_file_version,
                                img_file_path,
                            )

                    for (
                        img_file_version_tuple
                    ) in img_file_highest_version_dict.values():
                        img_file_path = img_file_version_tuple[1]

                        geo_file_name = img_file_path.name.replace("l2b", "l3bx")
                        geo_file_path = (
                            geo_file_orbit_dirs_path / orbit_dir_name / geo_file_name
                        )

                        if not geo_file_path.is_file():
                            continue

                        new_img_file_path = (
                            wavelength_output_dir_path
                            / img_file_path.relative_to(temp_output_dir_path)
                        )
                        new_img_file_path.parent.mkdir(parents=True, exist_ok=True)

                        new_geo_file_path = (
                            wavelength_output_dir_path
                            / geo_file_path.relative_to(temp_output_dir_path)
                        )
                        new_geo_file_path.parent.mkdir(parents=True, exist_ok=True)

                        # Move the files to their final wavelength dataset dirs
                        img_file_path.rename(new_img_file_path)
                        geo_file_path.rename(new_geo_file_path)


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
