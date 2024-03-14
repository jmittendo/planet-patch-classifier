import json
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from torch.utils import data

import source.config as config
import source.neural_network.config as nn_config
import source.neural_network.utility as nn_util
import source.patch_dataset.dataset as pd_dataset
import user.config as user_config
from source.neural_network.models import AutoencoderModel, SimCLREncoderModel
from source.neural_network.typing import (
    AutoencoderTrainParams,
    SimCLREncoderTrainParams,
)

if user_config.ENABLE_TEX_PLOTS:
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

    dataset = pd_dataset.get(name=dataset_name, version_name=version_name)
    train_dataset, test_dataset = data.random_split(
        dataset, [1 - user_config.TRAIN_TEST_RATIO, user_config.TRAIN_TEST_RATIO]
    )
    transforms = nn_util.get_patch_dataset_transforms(dataset)

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
                "loss_temperature": nn_config.SIMCLR_LOSS_TEMPERATURE,
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

    file_name_base = (
        f"{model_type}_{base_model_type}_{dataset.name}_{dataset.version_name}"
    )

    output_dir_path = nn_config.CHECKPOINTS_DIR_PATH / model_type
    output_dir_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir_path / f"{file_name_base}.pt"
    model.save_checkpoint(checkpoint_path)

    loss_table_path = output_dir_path / f"{file_name_base}_losses.pkl"
    loss_table_dict = {"train_losses": train_losses, "test_losses": test_losses}
    loss_table = DataFrame(data=loss_table_dict)
    loss_table.to_pickle(loss_table_path)

    model_info_json_path = output_dir_path / f"{file_name_base}_info.json"
    model_info_dict: dict[str, float | int] = train_params.copy()  # type: ignore
    model_info_dict.pop("output_interval")
    model_info_dict["best_epoch"] = best_epoch
    model_info_dict["best_test_loss"] = test_losses[best_epoch - 1]

    with open(model_info_json_path, "w") as model_info_json:
        json.dump(
            model_info_dict,
            model_info_json,
            indent=user_config.JSON_INDENT,
        )

    plots_dir_path = config.PLOTS_DIR_PATH / PLOTS_DIR_NAME
    plots_dir_path.mkdir(parents=True, exist_ok=True)

    plot_losses(
        train_losses,
        test_losses,
        best_epoch,
        plots_dir_path / f"{file_name_base}_losses.pdf",
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
