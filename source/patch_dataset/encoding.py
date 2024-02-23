import typing

from numpy import ndarray
from torch import no_grad

import source.neural_network.utility as nn_util
import source.patch_dataset.config as pd_config
from source.neural_network.models import SimpleEncoderModel

if typing.TYPE_CHECKING:
    from source.patch_dataset.dataset import PatchDataset


@no_grad()
def encode_dataset(dataset: "PatchDataset") -> ndarray:
    print("\nEncoding dataset...")

    transforms = nn_util.get_patch_dataset_transforms(dataset)
    encoder = SimpleEncoderModel(transforms=transforms)
    encoded_dataset = encoder.encode_dataset(
        dataset, batch_size=pd_config.ENCODING_BATCH_SIZE
    )

    return encoded_dataset.numpy()
