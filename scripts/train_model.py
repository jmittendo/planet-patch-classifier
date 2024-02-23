from argparse import ArgumentParser, Namespace

from torch.utils import data

import source.neural_network.config as nn_config
import source.neural_network.utility as nn_util
import source.patch_dataset.dataset as pd_dataset
import user.config as user_config
from source.neural_network.models import AutoencoderModel, SimCLREncoderModel
from source.neural_network.typing import (
    AutoencoderTrainParams,
    SimCLREncoderTrainParams,
)


def main() -> None:
    input_args = parse_input_args()
    model_type: str | None = input_args.model
    dataset_name: str | None = input_args.dataset
    version_name: str | None = input_args.version
    device: str | None = input_args.device

    if model_type is None:
        model_type = input("\nEnter model type ('autoencoder', 'simclr'): ")

    dataset = pd_dataset.get(name=dataset_name, version_name=version_name)
    train_dataset, test_dataset = data.random_split(dataset, [0.8, 0.2])
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
            model = SimCLREncoderModel(transforms=transforms)
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
    model.train(train_dataset, test_dataset, train_params)  # type: ignore

    checkpoint_path = (
        nn_config.CHECKPOINTS_DIR_PATH
        / model_type
        / f"{model_type}_{dataset.name}_{dataset.version_name}.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model.save_checkpoint(checkpoint_path)


def parse_input_args() -> Namespace:
    arg_parser = ArgumentParser(
        description=("Train the specified type of model on the specified dataset."),
    )

    arg_parser.add_argument("model", nargs="?", help="type of model")
    arg_parser.add_argument("dataset", nargs="?", help="name of the dataset")
    arg_parser.add_argument("version", nargs="?", help="version of the dataset to use")
    arg_parser.add_argument("-d", "--device", help="device to train the model on")

    return arg_parser.parse_args()


if __name__ == "__main__":
    main()
