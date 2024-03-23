import json
import typing

from pandas import DataFrame
from torch.nn import Module
from torchvision.transforms import Normalize, Resize

import source.neural_network.config as nn_config
import user.config as user_config
from source.neural_network.transforms import GrayscaleToRGB

if typing.TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from source.neural_network.models import EncoderModel
    from source.patch_dataset import PatchDataset


def get_patch_dataset_transforms(dataset: "PatchDataset") -> list[Module]:
    transforms: list[Module] = [Resize(224, antialias=True)]  # type: ignore

    if dataset.num_channels == 1:
        transforms.append(GrayscaleToRGB())
        normalize = Normalize(0.4589225, 0.2255861)
    else:
        normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    transforms.insert(0, normalize)

    return transforms


def get_checkpoint_path(
    model_type: str, base_model_type: str, dataset_name: str, dataset_version: str
) -> "Path":
    checkpoint_name = (
        f"{model_type}_{base_model_type}_{dataset_name}_{dataset_version}.pt"
    )

    return nn_config.CHECKPOINTS_DIR_PATH / model_type / checkpoint_name


def save_checkpoint(
    model: "EncoderModel",
    model_type: str,
    base_model_type: str,
    dataset_name: str,
    dataset_version: str,
    train_losses: list[float],
    test_losses: list[float],
    best_epoch: int,
    train_params: dict[str, "Any"],
) -> "Path":
    checkpoint_path = get_checkpoint_path(
        model_type, base_model_type, dataset_name, dataset_version
    )

    parent_dir_path = checkpoint_path.parent
    parent_dir_path.mkdir(parents=True, exist_ok=True)

    model.save_checkpoint(checkpoint_path)

    checkpoint_stem = checkpoint_path.stem

    loss_table_path = parent_dir_path / f"{checkpoint_stem}_losses.pkl"
    loss_table_dict = {"train_losses": train_losses, "test_losses": test_losses}
    loss_table = DataFrame(data=loss_table_dict)
    loss_table.to_pickle(loss_table_path)

    model_info_json_path = parent_dir_path / f"{checkpoint_stem}_info.json"
    model_info_dict: dict[str, float | int] = train_params.copy()
    model_info_dict.pop("output_interval")
    model_info_dict["best_epoch"] = best_epoch
    model_info_dict["best_test_loss"] = test_losses[best_epoch - 1]

    with open(model_info_json_path, "w") as model_info_json:
        json.dump(model_info_dict, model_info_json, indent=user_config.JSON_INDENT)

    return checkpoint_path
