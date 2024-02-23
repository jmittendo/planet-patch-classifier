from typing import TypeAlias, TypedDict

import torch


class SimCLREncoderTrainParams(TypedDict):
    batch_size: int
    loss_temperature: float
    base_learning_rate: float
    epochs: int
    output_interval: int


class AutoencoderTrainParams(TypedDict):
    batch_size: int
    learning_rate: float
    epochs: int
    output_interval: int


DeviceLike: TypeAlias = str | torch.device
