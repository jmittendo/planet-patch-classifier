import typing

import torch
from numpy import ndarray
from torch import no_grad
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights
from torch.nn import Identity

import source.patch_dataset.config as pd_config

if typing.TYPE_CHECKING:
    from source.patch_dataset.dataset import PatchDataset


@no_grad()
def encode_dataset(dataset: "PatchDataset") -> ndarray:
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = Identity()  # type: ignore

    data_loader = DataLoader(dataset, batch_size=pd_config.ENCODING_BATCH_SIZE)
    encoded_tensors = [model(batch_tensor) for batch_tensor in data_loader]

    return torch.cat(encoded_tensors).numpy()
