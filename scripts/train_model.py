# This file is part of planet-patch-classifier, a Python tool for generating and
# classifying planet patches from satellite imagery via unsupervised machine learning
# Copyright (C) 2024  Jan Mittendorf (jmittendo)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data

import source.neural_network as nn
import source.patch_dataset as pd
import user.config as user_config
from source import config
from source.neural_network import (
    AutoencoderModel,
    AutoencoderTrainParams,
    SimCLREncoderModel,
    SimCLREncoderTrainParams,
)

if user_config.PLOT_ENABLE_TEX:
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.titlepad"] = 8

PLOTS_DIR_NAME = "train-losses"


def main() -> None:
    input_args = parse_input_args()
    model_type: str | None = input_args.model
    base_model_type: str | None = input_args.base_model
    dataset_name: str | None = input_args.dataset
    version_name: str | None = input_args.version
    device: str | None = input_args.device

    if model_type is None:
        model_type = input("\nEnter model type ('autoencoder', 'simclr'): ")

    if model_type != "autoencoder" and base_model_type is None:
        base_model_type = input(
            "Enter encoder base model type "
            "('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'): "
        )

    if model_type == "autoencoder" and base_model_type != "resnet18":
        warnings.warn("Autoencoder model always uses 'resnet18' base model")
        base_model_type = "resnet18"

    dataset = pd.get_dataset(name=dataset_name, version_name=version_name)
    train_dataset, test_dataset = data.random_split(
        dataset, [1 - user_config.TRAIN_TEST_RATIO, user_config.TRAIN_TEST_RATIO]
    )
    transforms = nn.get_patch_dataset_transforms(dataset)

    match model_type:
        case "autoencoder":
            model = AutoencoderModel(transforms=transforms)
            train_params: AutoencoderTrainParams = {  # type: ignore
                "batch_size": user_config.TRAIN_BATCH_SIZE,
                "learning_rate": 0.3 * user_config.TRAIN_BATCH_SIZE / 256,  # type: ignore
                "epochs": user_config.TRAIN_EPOCHS,
                "output_interval": user_config.TRAIN_OUTPUT_INTERVAL,
            }
        case "simclr":
            model = SimCLREncoderModel(base_model_type, transforms=transforms)  # type: ignore
            train_params: SimCLREncoderTrainParams = {
                "batch_size": user_config.TRAIN_BATCH_SIZE,
                "loss_temperature": user_config.SIMCLR_LOSS_TEMPERATURE,
                "base_learning_rate": 0.3 * user_config.TRAIN_BATCH_SIZE / 256,
                "epochs": user_config.TRAIN_EPOCHS,
                "output_interval": user_config.TRAIN_OUTPUT_INTERVAL,
            }
        case _:
            raise ValueError(f"'{model_type}' is not a valid model type")

    if device is not None:
        model.move_to_device(device)

    print()
    train_losses, test_losses, best_epoch = model.train(
        train_dataset, test_dataset, train_params  # type: ignore
    )

    checkpoint_path = nn.save_checkpoint(
        model,
        model_type,
        base_model_type,  # type: ignore
        dataset.name,
        dataset.version_name,
        train_losses,
        test_losses,
        best_epoch,
        train_params,  # type: ignore
    )

    plots_dir_path = config.PLOTS_DIR_PATH / PLOTS_DIR_NAME
    plots_dir_path.mkdir(parents=True, exist_ok=True)

    plot_losses(
        train_losses,
        test_losses,
        best_epoch,
        plots_dir_path / f"{checkpoint_path.stem}_losses.pdf",
    )


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=("Train the specified type of model on the specified dataset."),
    )

    arg_parser.add_argument("model", nargs="?", help="type of model")
    arg_parser.add_argument("base_model", nargs="?", help="type of base model")
    arg_parser.add_argument("dataset", nargs="?", help="name of the dataset")
    arg_parser.add_argument("version", nargs="?", help="version of the dataset to use")
    arg_parser.add_argument("-d", "--device", help="device to train the model on")

    return arg_parser.parse_args()


def plot_losses(
    train_losses: list[float],
    test_losses: list[float],
    best_epoch: int,
    file_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))

    epoch_values = (np.arange(len(train_losses)) + 1).astype(int)

    ax.plot(epoch_values, train_losses, label="Train loss")
    ax.plot(epoch_values, test_losses, label="Test loss")
    ax.axvline(best_epoch, label="Best test loss", linestyle="--", color="black")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fancybox=False)
    ax.set_xlim(epoch_values.min(), epoch_values.max())

    fig.savefig(file_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
