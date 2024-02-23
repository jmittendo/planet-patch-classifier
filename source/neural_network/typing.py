from typing import TypedDict


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
