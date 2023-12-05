import warnings
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

DIRECTORY_URL = "https://archives.esac.esa.int/psa/ftp/VENUS-EXPRESS/VMC/"
SUB_URL_FILTER = "VEX-V-VMC-3-RDR"
FILE_NAME_FILTER = "UV2"
OUTPUT_DIR = "datasets/satellite-datasets/vex-test"


def main():
    dir_url_stripped = DIRECTORY_URL.rstrip("/")

    mission_dir_names = [
        href.rstrip("/")
        for href in get_hrefs(
            DIRECTORY_URL, dir_only=True, href_filters=[SUB_URL_FILTER]
        )
    ]

    for mission_dir_name in mission_dir_names:
        mission_dir_url = f"{dir_url_stripped}/{mission_dir_name}"

        print("Retrieving image file urls... (1 / 2)")
        image_file_urls = get_data_dir_file_urls(
            mission_dir_url, "DATA", FILE_NAME_FILTER, ".IMG"
        )

        print("Retrieving geometry file urls... (2 / 2)")
        geometry_file_urls = get_data_dir_file_urls(
            mission_dir_url, "GEOMETRY", FILE_NAME_FILTER, ".GEO"
        )

        for image_file_url in image_file_urls:
            print(image_file_url)


def get_data_dir_file_urls(
    mission_dir_url: str, data_dir_name: str, file_name_filter: str, file_extension: str
) -> list[str]:
    data_dir_url = f"{mission_dir_url}/{data_dir_name}"

    data_orbit_dir_names = [
        href.rstrip("/") for href in get_hrefs(data_dir_url, dir_only=True)
    ]

    file_urls: list[str] = []

    for orbit_dir_name in tqdm(data_orbit_dir_names):
        orbit_dir_url = f"{data_dir_url}/{orbit_dir_name}"

        file_names = get_hrefs(
            orbit_dir_url, href_filters=[file_name_filter, file_extension]
        )

        for file_name in file_names:
            file_url = f"{orbit_dir_url}/{file_name}"
            file_urls.append(file_url)

    return file_urls


def get_hrefs(
    url: str, dir_only: bool = False, href_filters: list[str] | None = None
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

        if href_filters is not None and not all(hf in href for hf in href_filters):
            continue

        hrefs.append(href)

    return hrefs


if __name__ == "__main__":
    main()
