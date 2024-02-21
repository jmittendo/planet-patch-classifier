import typing

import torch
from numpy import ndarray
from torch import no_grad
from torch.nn import Identity, Module
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision.transforms import Compose, Normalize, Resize

import source.patch_dataset.config as pd_config
from source.neural_network.transforms import GrayscaleToRGB

if typing.TYPE_CHECKING:
    from source.patch_dataset.dataset import PatchDataset


@no_grad()
def encode_dataset(dataset: "PatchDataset") -> ndarray:
    print("Encoding dataset...")

    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = Identity()  # type: ignore
    model.eval()

    transforms_list: list[Module] = [Resize(224, antialias=True)]  # type: ignore

    if dataset.num_channels == 1:
        transforms_list.append(GrayscaleToRGB())
        normalize = Normalize(0.4589225, 0.2255861)
    else:
        normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    transforms_list.insert(0, normalize)

    transforms = Compose(transforms_list)
    data_loader = DataLoader(dataset, batch_size=pd_config.ENCODING_BATCH_SIZE)
    encoded_tensors = [model(transforms(batch_tensor)) for batch_tensor in data_loader]

    return torch.cat(encoded_tensors).numpy()
