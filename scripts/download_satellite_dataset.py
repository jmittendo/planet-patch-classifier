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
from argparse import ArgumentParser, Namespace

import source.satellite_dataset.config as sdcfg
import source.satellite_dataset.download as sd_download
import source.utility as util
from source.satellite_dataset.typing import DownloadConfig


def main() -> None:
    util.configure_logging("download")

    with open(sdcfg.DATASET_DOWNLOADS_JSON_PATH, "r") as json_file:
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

    sd_download.download_dataset(download_config, dataset_name)


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=(
            "Download a named dataset. Run without arguments to get a list of"
            "available datasets."
        ),
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
