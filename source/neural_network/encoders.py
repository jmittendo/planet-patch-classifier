import abc

import torch
from torch import Tensor
from torch.nn import Identity, Module
from torchvision.models import ResNet18_Weights, resnet18

from source.neural_network.modules import DecoderResNet18, EncoderResNet18


class Encoder(Module):
    @abc.abstractmethod
    def encode(self, input_tensor: Tensor) -> Tensor:
        raise NotImplementedError


class Autoencoder(Encoder):
    def __init__(self, image_channels: int = 3, image_resolution: int = 224) -> None:
        super().__init__()

        self._encoder_resnet = EncoderResNet18(image_channels=image_channels)
        self._decoder_resnet = DecoderResNet18(
            output_resolution=image_resolution, output_channels=image_channels
        )

    def encode(self, input_tensor: Tensor) -> Tensor:
        return self._encode(input_tensor, return_maxpool_indices=False)  # type: ignore

    def _encode(
        self, input_tensor: Tensor, return_maxpool_indices: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        return self._encoder_resnet(
            input_tensor, return_maxpool_indices=return_maxpool_indices
        )

    def _decode(self, input_tensor: Tensor, maxpool_indices: Tensor) -> Tensor:
        return self._decoder_resnet(input_tensor, maxpool_indices)

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self._decode(*self._encode(input_tensor, return_maxpool_indices=True))


class SimCLREncoder(Encoder):
    # See: https://arxiv.org/abs/2002.05709

    def __init__(self) -> None:
        super().__init__()

        self._base_encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self._base_encoder.fc = Identity()  # type: ignore

        self._projection_head = torch.nn.Sequential(
            torch.nn.Linear(512, 256, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64, bias=True),
        )

    def encode(self, input_tensor: Tensor) -> Tensor:
        return self._base_encoder(input_tensor)

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self._projection_head(self._base_encoder(input_tensor))


class SimpleEncoder(Encoder):
    def __init__(self) -> None:
        super().__init__()

        self._encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self._encoder.fc = Identity()  # type: ignore

    def encode(self, input_tensor: Tensor) -> Tensor:
        return self(input_tensor)

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self._encoder(input_tensor)
