import logging
import shutil
import warnings
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag
from requests import HTTPError
from tqdm import tqdm

import source.satellite_dataset.config as sd_config
import user.config as user_config
from source.satellite_dataset.typing import DownloadConfig


def download_dataset(download_config: DownloadConfig, dataset_name: str) -> None:
    dataset_archive = download_config["archive"]
    dataset_instrument = download_config["instrument"]
    dataset_wavelengths = download_config["wavelengths"]

    output_dir_path = sd_config.DATASET_DOWNLOADS_DIR_PATH / dataset_name

    match dataset_archive:
        case "vex-vmc":
            _download_vex_vmc_dataset(
                dataset_wavelengths[0],
                output_dir_path,
                user_config.DOWNLOAD_CHUNK_SIZE,
            )
        case "vco":
            _download_vco_dataset(
                dataset_instrument,
                dataset_wavelengths,
                dataset_name,
                output_dir_path,
                user_config.DOWNLOAD_CHUNK_SIZE,
            )
        case _:
            raise ValueError(
                "No download script implemented for dataset archive "
                f"'{dataset_archive}'"
            )


def _download_vex_vmc_dataset(
    wavelength_filter: str, output_dir_path: Path, chunk_size: int
) -> None:
    archive_url = "https://archives.esac.esa.int/psa/ftp/VENUS-EXPRESS/VMC/"
    archive_url_stripped = archive_url.rstrip("/")

    href_filter = "VEX-V-VMC-3-RDR"

    mission_dir_names = [
        href.rstrip("/")
        for href in _get_hrefs(archive_url, dir_only=True)
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
            for href in _get_hrefs(img_dir_url, dir_only=True)
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
                for href in _get_hrefs(orbit_dir_url)
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
                    _download_file(
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
                    _download_file(
                        img_file_url,
                        img_file_path,
                        chunk_size=chunk_size,
                        pbar_indent=2,
                    )
                except HTTPError:
                    warnings.warn(
                        "Error while trying to download image file at URL "
                        f"'{img_file_url}'"
                    )


def _download_vco_dataset(
    instrument: str,
    wavelength_filters: list[str],
    dataset_name: str,
    temp_output_dir_path: Path,
    chunk_size: int,
) -> None:
    logging.debug(
        "Starting VCO dataset download with parameters: "
        f"{instrument = }, {wavelength_filters = }, {dataset_name = }, {chunk_size = }"
    )

    archive_url = "https://data.darts.isas.jaxa.jp/pub/pds3/"
    archive_url_stripped = archive_url.rstrip("/")

    logging.debug(f"{temp_output_dir_path = }")

    wavelength_output_dir_paths = {
        wavelength_filter: (
            temp_output_dir_path.with_name(f"{dataset_name}-{wavelength_filter}")
        )
        for wavelength_filter in wavelength_filters
    }

    img_href_filter = f"vco-v-{instrument}-3-cdr-v"

    img_file_dir_names = [
        href.rstrip("/")
        for href in _get_hrefs(archive_url_stripped, dir_only=True)
        if img_href_filter in href
    ]

    logging.debug("Looping over image file dir names")
    for img_file_dir_name in tqdm(
        img_file_dir_names, desc="Image files download progress", leave=False
    ):
        logging.debug(f"{img_file_dir_name = }")

        img_file_dir_version_str = img_file_dir_name.split("-")[5]

        img_file_dir_url = f"{archive_url_stripped}/{img_file_dir_name}"
        logging.debug(f"{img_file_dir_url = }")

        img_file_dir_path = temp_output_dir_path / img_file_dir_name
        img_file_dir_path.mkdir(parents=True, exist_ok=True)
        logging.debug(f"{img_file_dir_path = }")

        geo_file_dir_name = f"vco_{instrument}_l3_{img_file_dir_version_str}"
        geo_file_dir_url = f"{archive_url_stripped}/extras/{geo_file_dir_name}"
        logging.debug(f"{geo_file_dir_url = }")

        geo_file_dir_path = temp_output_dir_path / "extras" / geo_file_dir_name
        geo_file_dir_path.mkdir(parents=True, exist_ok=True)
        logging.debug(f"{geo_file_dir_path = }")

        img_zip_file_names = [
            href
            for href in _get_hrefs(img_file_dir_url)
            if (".zip" in href or ".tar.gz" in href or ".tar.xz" in href)
            and "netcdf" not in href
        ]

        logging.debug("Looping over image zip file names")
        for img_zip_file_name in tqdm(
            img_zip_file_names,
            desc=f"└ Image file directory '{img_file_dir_name}' progress",
            leave=False,
        ):
            logging.debug(f"{img_zip_file_name = }")

            img_zip_file_url = f"{img_file_dir_url}/{img_zip_file_name}"
            logging.debug(f"{img_zip_file_url = }")

            img_zip_file_path = img_file_dir_path / img_zip_file_name
            logging.debug(f"{img_zip_file_path = }")

            geo_zip_file_name_stem = (
                f"{img_zip_file_name.split('.')[0].replace('_1', '_7')}_l3x_fits"
            )
            logging.debug(f"{geo_zip_file_name_stem = }")

            # If geometry zip file (url) does not exist skip both downloads
            # NOTE: Considering the code above, this can happen when an image zip file
            # does not have a corresponding geometry zip file and is therefore not a
            # critical error.
            geo_zip_file_path: Path | None = None

            logging.debug("Looping over possible geometry zip file suffixes")
            for geo_zip_file_suffix in [".zip", ".tar.gz", ".tar.xz"]:
                logging.debug(f"{geo_zip_file_suffix = }")

                geo_zip_file_name = geo_zip_file_name_stem + geo_zip_file_suffix
                geo_zip_file_url = f"{geo_file_dir_url}/{geo_zip_file_name}"
                geo_zip_file_path = geo_file_dir_path / geo_zip_file_name

                try:
                    logging.debug(
                        "Attempting download of geometry zip file URL: "
                        f"'{geo_zip_file_url}' to path '{geo_zip_file_path}'"
                    )
                    _download_file(
                        geo_zip_file_url,
                        geo_zip_file_path,
                        chunk_size=chunk_size,
                        pbar_indent=1,
                    )
                    break
                except HTTPError:
                    logging.debug(
                        f"Download of geometry zip file URL: {geo_zip_file_url}"
                        "failed, continuing loop..."
                    )
                    geo_zip_file_path = None
                    continue

            if geo_zip_file_path is None:
                logging.debug("No geometry zip file found, continuing loop...")
                continue

            # If image zip file (url) does not exist skip download
            # NOTE: Unlike in the case of the geometry zip files this should never
            # happen as per the code above. However, if for some unforeseen reason
            # it happens anyway, we do not want to stop the whole download process
            # and simply warn the user.
            try:
                logging.debug(
                    "Attempting download of image zip file URL: "
                    f"'{img_zip_file_url}' to path: '{img_zip_file_path}'"
                )
                _download_file(
                    img_zip_file_url,
                    img_zip_file_path,
                    chunk_size=chunk_size,
                    pbar_indent=1,
                )
            except HTTPError:
                warnings.warn(
                    f"Error while trying to download image zip file at URL"
                    f"'{img_zip_file_url}'"
                )
                logging.debug(
                    f"Error while trying to download image zip file URL: "
                    f"'{img_zip_file_url}', continuing loop..."
                )
                continue

            # Unpack the zip files with automatic format detection and delete them after
            logging.debug(
                f"Unpacking image zip file '{img_zip_file_path}' to dir "
                f"'{img_file_dir_path}'"
            )
            shutil.unpack_archive(img_zip_file_path, extract_dir=img_file_dir_path)

            logging.debug(f"Deleting image zip file '{img_zip_file_path}'")
            img_zip_file_path.unlink(missing_ok=True)

            logging.debug(
                f"Unpacking geometry zip file '{geo_zip_file_path}' to dir "
                f"'{geo_file_dir_path}'"
            )
            shutil.unpack_archive(geo_zip_file_path, extract_dir=geo_file_dir_path)

            logging.debug(f"Deleting geometry zip file '{geo_zip_file_path}'")
            geo_zip_file_path.unlink(missing_ok=True)

            unpacked_img_file_dir_name = "_".join(img_zip_file_path.stem.split("_")[:2])
            unpacked_geo_file_dir_name = "_".join(geo_zip_file_path.stem.split("_")[:2])

            img_file_orbit_dirs_path = (
                img_file_dir_path / unpacked_img_file_dir_name / "data" / "l2b"
            )
            logging.debug(f"{img_file_orbit_dirs_path = }")

            geo_file_orbit_dirs_path = (
                geo_file_dir_path
                / unpacked_geo_file_dir_name
                / "data"
                / "l3bx"
                / "fits"
            )
            logging.debug(f"{geo_file_orbit_dirs_path = }")

            logging.debug("Looping over image file orbit dirs")
            for img_file_orbit_dir_path in img_file_orbit_dirs_path.iterdir():
                logging.debug(f"{img_file_orbit_dir_path = }")

                if not img_file_orbit_dir_path.is_dir():
                    logging.debug(
                        f"'{img_file_orbit_dir_path}' is not a directory, "
                        "continuing loop..."
                    )
                    continue

                orbit_dir_name = img_file_orbit_dir_path.name
                logging.debug(f"{orbit_dir_name = }")

                logging.debug("Looping over wavelength filters")
                for wavelength_filter in wavelength_filters:
                    logging.debug(f"{wavelength_filter = }")

                    wavelength_output_dir_path = wavelength_output_dir_paths[
                        wavelength_filter
                    ]
                    logging.debug(f"{wavelength_output_dir_path = }")

                    # Find only the highest version for each file and save its path
                    img_file_highest_version_dict: dict[str, tuple[int, Path]] = {}

                    logging.debug("Looping over wavelength image files in orbit dir")
                    for img_file_path in img_file_orbit_dir_path.glob(
                        f"*{wavelength_filter}*.fit"
                    ):
                        logging.debug(f"{img_file_path = }")

                        img_file_stem_components = img_file_path.stem.split("_")
                        img_file_name_base = "_".join(img_file_stem_components[:5])
                        logging.debug(f"{img_file_name_base = }")

                        img_file_version = int(img_file_stem_components[5].lstrip("v"))
                        logging.debug(f"{img_file_version = }")

                        current_highest_version_tuple = (
                            img_file_highest_version_dict.get(img_file_name_base)
                        )
                        current_highest_version = (
                            current_highest_version_tuple[0]
                            if current_highest_version_tuple is not None
                            else 0
                        )

                        if img_file_version > current_highest_version:
                            logging.debug(
                                f"Found new highest version: '{img_file_version}' "
                                f"(Previous: '{current_highest_version}')"
                            )

                            img_file_highest_version_dict[img_file_name_base] = (
                                img_file_version,
                                img_file_path,
                            )

                    logging.debug("Looping over highest version files")
                    for (
                        img_file_version_tuple
                    ) in img_file_highest_version_dict.values():
                        img_file_path = img_file_version_tuple[1]
                        logging.debug(f"{img_file_path = }")

                        geo_file_name = img_file_path.name.replace("l2b", "l3bx")
                        geo_file_path = (
                            geo_file_orbit_dirs_path / orbit_dir_name / geo_file_name
                        )
                        logging.debug(f"{geo_file_path = }")

                        if not geo_file_path.is_file():
                            logging.debug(
                                f"Geometry file '{geo_file_path}' not found, "
                                "continuing loop..."
                            )
                            continue

                        new_img_file_path = (
                            wavelength_output_dir_path
                            / img_file_path.relative_to(temp_output_dir_path)
                        )
                        new_img_file_path.parent.mkdir(parents=True, exist_ok=True)
                        logging.debug(f"{new_img_file_path = }")

                        new_geo_file_path = (
                            wavelength_output_dir_path
                            / geo_file_path.relative_to(temp_output_dir_path)
                        )
                        new_geo_file_path.parent.mkdir(parents=True, exist_ok=True)
                        logging.debug(f"{new_geo_file_path = }")

                        # Move the files to their final wavelength dataset dirs
                        logging.debug("Moving image and geometry files to final dirs")
                        img_file_path.rename(new_img_file_path)
                        geo_file_path.rename(new_geo_file_path)

                    logging.debug("Finished loop over highest version files")
                logging.debug("Finished loop over wavelength filters")
            logging.debug("Finished loop over image file orbit dirs")
        logging.debug("Finished loop over image zip file names")
    logging.debug("Finished loop over image file dir names")

    # Delete temporary download directory once downloads are finished
    logging.debug(f"Deleting temporary output dir: '{temp_output_dir_path}'")
    shutil.rmtree(temp_output_dir_path)

    logging.debug(
        "Finished VCO dataset download with parameters: "
        f"{instrument = }, {wavelength_filters = }, {dataset_name = }, {chunk_size = }"
    )


def _get_hrefs(url: str, dir_only: bool = False) -> list[str]:
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


def _download_file(
    file_url: str, file_path: Path, *, chunk_size: int, pbar_indent: int
):
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
