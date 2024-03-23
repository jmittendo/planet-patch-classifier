import typing
import warnings
from pathlib import Path

from numpy import ndarray
from torch import no_grad

import source.neural_network as nn
import source.patch_dataset.config as pd_config
from source.neural_network import (
    AutoencoderModel,
    SimCLREncoderModel,
    SimpleEncoderModel,
)
from source.neural_network.typing import DeviceLike

if typing.TYPE_CHECKING:
    from source.patch_dataset import PatchDataset


@no_grad()
def encode_dataset(
    dataset: "PatchDataset",
    model: str,
    base_model: str,
    checkpoint_path: Path | None = None,
    device: DeviceLike | None = None,
) -> ndarray:
    print("\nEncoding dataset...")

    if model in ["autoencoder", "simclr"] and checkpoint_path is None:
        warnings.warn(f"No model checkpoint selected with model type '{model}'")

    if model == "simple" and checkpoint_path is not None:
        warnings.warn(f"Model checkpoint not used with model type '{model}'")

    transforms = nn.get_patch_dataset_transforms(dataset)

    match model:
        case "simple":
            encoder = SimpleEncoderModel(base_model=base_model, transforms=transforms)
        case "autoencoder":
            if base_model != "resnet18":
                warnings.warn("Autoencoder model always uses 'resnet18' base model")

            encoder = AutoencoderModel(
                transforms=transforms, checkpoint_path=checkpoint_path
            )
        case "simclr":
            encoder = SimCLREncoderModel(
                base_model=base_model,
                transforms=transforms,
                checkpoint_path=checkpoint_path,
            )
        case _:
            raise ValueError(f"'{model}' is not a valid model type")

    if device is not None:
        encoder.move_to_device(device)

    encoded_dataset = encoder.encode_dataset(
        dataset, batch_size=pd_config.ENCODING_BATCH_SIZE
    )

    return encoded_dataset.numpy()
