import warnings
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

DIRECTORY_URL = "https://archives.esac.esa.int/psa/ftp/VENUS-EXPRESS/VMC/"
SUB_URL_FILTER = "VEX-V-VMC-3-RDR"
FILE_NAME_FILTER = "UV2"
OUTPUT_DIR = "datasets/satellite-datasets/vex-test"


def main() -> None:
    dir_url_stripped = DIRECTORY_URL.rstrip("/")

    mission_dir_names = [
        href.rstrip("/")
        for href in get_hrefs(
            DIRECTORY_URL, dir_only=True, positive_filters=[SUB_URL_FILTER]
        )
    ]

    for mission_dir_name in tqdm(
        mission_dir_names, desc="Full download progress", leave=False
    ):
        mission_dir_url = f"{dir_url_stripped}/{mission_dir_name}"
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
                orbit_dir_url, positive_filters=[FILE_NAME_FILTER, ".IMG"]
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
