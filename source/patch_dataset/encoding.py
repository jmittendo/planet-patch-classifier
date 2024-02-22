import typing

from numpy import ndarray
from torch import no_grad
from torch.nn import Module
from torchvision.transforms import Normalize, Resize

import source.patch_dataset.config as pd_config
from source.neural_network.models import SimpleEncoderModel
from source.neural_network.transforms import GrayscaleToRGB

if typing.TYPE_CHECKING:
    from source.patch_dataset.dataset import PatchDataset


@no_grad()
def encode_dataset(dataset: "PatchDataset") -> ndarray:
    print("\nEncoding dataset...")

    transforms_list: list[Module] = [Resize(224, antialias=True)]  # type: ignore

    if dataset.num_channels == 1:
        transforms_list.append(GrayscaleToRGB())
        normalize = Normalize(0.4589225, 0.2255861)
    else:
        normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    transforms_list.insert(0, normalize)

    encoder = SimpleEncoderModel(transforms=transforms_list)
    encoded_dataset = encoder.encode_dataset(
        dataset, batch_size=pd_config.ENCODING_BATCH_SIZE
    )

    return encoded_dataset.numpy()
