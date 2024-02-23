from torch.nn import Module
from torchvision.transforms import Normalize, Resize

from source.neural_network.transforms import GrayscaleToRGB
from source.patch_dataset.dataset import PatchDataset


def get_patch_dataset_transforms(dataset: PatchDataset) -> list[Module]:
    transforms: list[Module] = [Resize(224, antialias=True)]  # type: ignore

    if dataset.num_channels == 1:
        transforms.append(GrayscaleToRGB())
        normalize = Normalize(0.4589225, 0.2255861)
    else:
        normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    transforms.insert(0, normalize)

    return transforms
