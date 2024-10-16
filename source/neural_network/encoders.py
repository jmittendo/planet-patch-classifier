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

import abc

import torch
from torch import Tensor
from torch.nn import Identity, Module
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)

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

    def __init__(self, base_model: str) -> None:
        super().__init__()

        match base_model:
            case "resnet18":
                self._base_encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            case "resnet34":
                self._base_encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            case "resnet50":
                self._base_encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            case "resnet101":
                self._base_encoder = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
            case "resnet152":
                self._base_encoder = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
            case _:
                raise ValueError(f"'{base_model}' is not a valid base model")

        final_layer_size = self._base_encoder.fc.in_features

        self._base_encoder.fc = Identity()  # type: ignore

        self._projection_head = torch.nn.Sequential(
            torch.nn.Linear(final_layer_size, final_layer_size // 2, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(final_layer_size // 2, final_layer_size // 8, bias=True),
        )

    def encode(self, input_tensor: Tensor) -> Tensor:
        return self._base_encoder(input_tensor)

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self._projection_head(self._base_encoder(input_tensor))


class SimpleEncoder(Encoder):
    def __init__(self, base_model: str) -> None:
        super().__init__()

        match base_model:
            case "resnet18":
                self._encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            case "resnet34":
                self._encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            case "resnet50":
                self._encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            case "resnet101":
                self._encoder = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
            case "resnet152":
                self._encoder = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
            case _:
                raise ValueError(f"'{base_model}' is not a valid base model")

        self._encoder.fc = Identity()  # type: ignore

    def encode(self, input_tensor: Tensor) -> Tensor:
        return self(input_tensor)

    def forward(self, input_tensor: Tensor) -> Tensor:
        return self._encoder(input_tensor)
