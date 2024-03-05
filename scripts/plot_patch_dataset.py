from argparse import ArgumentParser, Namespace

import source.neural_network.config as nn_config
import source.patch_dataset.dataset as pd_dataset


def main() -> None:
    input_args = parse_input_args()
    dataset_name: str | None = input_args.name
    version_name: str | None = input_args.version
    encoder_model: str | None = input_args.encoder
    num_patches: int | None = input_args.number

    dataset = pd_dataset.get(name=dataset_name, version_name=version_name)

    if encoder_model is None:
        encoder_model = input(
            "Enter encoder model type ('simple', 'autoencoder', 'simclr'): "
        )

    checkpoint_path = (
        None
        if encoder_model == "simple"
        else nn_config.CHECKPOINTS_DIR_PATH
        / encoder_model
        / f"{encoder_model}_{dataset.name}_{dataset.version_name}.pt"
    )

    if checkpoint_path is not None and not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"No model checkpoint found for encoder model type '{encoder_model}' and "
            f"dataset '{dataset.name}' with version '{dataset.version_name}'"
        )

    dataset.plot_geometry_scatter(num_patches=num_patches)
    dataset.plot_encoded_tsne_scatter(encoder_model, checkpoint_path=checkpoint_path)


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=("Generate a scatter plot for the specified dataset."),
    )

    arg_parser.add_argument("name", nargs="?", help="name of the dataset")
    arg_parser.add_argument(
        "version", nargs="?", help="version of the dataset to use for plotting"
    )
    arg_parser.add_argument("encoder", nargs="?", help="model type to use for encoding")
    arg_parser.add_argument(
        "-n",
        "--number",
        type=int,
        help="number of patches to plot for geometry scatter",
    )

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
