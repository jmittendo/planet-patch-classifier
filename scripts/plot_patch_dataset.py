import warnings
from argparse import ArgumentParser, Namespace

import source.neural_network as nn
import source.patch_dataset as pd


def main() -> None:
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name
    version_name: str | None = input_args.version
    encoder_model: str | None = input_args.encoder
    encoder_base_model: str | None = input_args.base_model
    num_patches: int | None = input_args.number

    dataset = pd.get_dataset(name=dataset_name, version_name=version_name)

    if encoder_model is None:
        encoder_model = input(
            "Enter encoder model type ('simple', 'autoencoder', 'simclr'): "
        )

    if encoder_model != "autoencoder" and encoder_base_model is None:
        encoder_base_model = input(
            "Enter encoder base model type "
            "('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'): "
        )

    if encoder_model == "autoencoder" and encoder_base_model != "resnet18":
        warnings.warn("Autoencoder model always uses 'resnet18' base model")
        encoder_base_model = "resnet18"

    checkpoint_path = (
        None
        if encoder_model == "simple"
        else nn.get_checkpoint_path(
            encoder_model, encoder_base_model, dataset.name, dataset.version_name  # type: ignore
        )
    )

    if checkpoint_path is not None and not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"No model checkpoint found for encoder model type '{encoder_model}' and "
            f"dataset '{dataset.name}' with version '{dataset.version_name}'"
        )

    dataset.plot_geometry_scatter(num_patches=num_patches)
    dataset.plot_encoded_tsne_scatter(
        encoder_model, encoder_base_model, checkpoint_path=checkpoint_path  # type: ignore
    )


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=("Generate a scatter plot for the specified dataset."),
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")
    arg_parser.add_argument(
        "version", nargs="?", help="version of the dataset to use for plotting"
    )
    arg_parser.add_argument("encoder", nargs="?", help="model type to use for encoding")
    arg_parser.add_argument("base_model", nargs="?", help="base model type for encoder")
    arg_parser.add_argument(
        "-n",
        "--number",
        type=int,
        help="number of patches to plot for geometry scatter",
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
