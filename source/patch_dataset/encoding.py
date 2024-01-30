import typing

import torch
from numpy import ndarray
from torch import Tensor, no_grad
from torch.nn import Identity, Module
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision.transforms import Normalize, Resize, Compose

import source.patch_dataset.config as pd_config

if typing.TYPE_CHECKING:
    from source.patch_dataset.dataset import PatchDataset


class GrayscaleToRGB(Module):
    def forward(self, input_tensor: Tensor):
        return torch.cat([input_tensor] * 3, dim=-3)


@no_grad()
def encode_dataset(dataset: "PatchDataset") -> ndarray:
    print("Encoding dataset...")

    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = Identity()  # type: ignore

    transforms_list = [
        Normalize(dataset.mean, dataset.std),
        Resize(224, antialias=True),  # type: ignore
        GrayscaleToRGB(),
    ]
    transforms = Compose(transforms_list)

    data_loader = DataLoader(dataset, batch_size=pd_config.ENCODING_BATCH_SIZE)

    encoded_tensors = [model(transforms(batch_tensor)) for batch_tensor in data_loader]

    return torch.cat(encoded_tensors).numpy()
