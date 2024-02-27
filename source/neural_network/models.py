from pathlib import Path

import torch
from torch import Tensor, no_grad
from torch.nn import Identity, Module, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    ColorJitter,
    Compose,
    GaussianBlur,
    InterpolationMode,
    RandomApply,
    RandomHorizontalFlip,
    RandomResizedCrop,
)
from tqdm import tqdm

from source.neural_network.encoders import (
    Autoencoder,
    Encoder,
    SimCLREncoder,
    SimpleEncoder,
)
from source.neural_network.losses import NTXentLoss
from source.neural_network.optimizers import LARS
from source.neural_network.transforms import RandomCroppedRotation
from source.neural_network.typing import (
    AutoencoderTrainParams,
    DeviceLike,
    SimCLREncoderTrainParams,
)


class EncoderModel:
    def __init__(
        self,
        encoder: Encoder,
        transforms: list[Module] | None = None,
        checkpoint_path: Path | None = None,
    ):
        self._encoder = encoder
        self._transforms = Identity() if transforms is None else Compose(transforms)
        self.current_device = torch.device("cpu")

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def __call__(self, input_tensor: Tensor) -> Tensor:
        return self._encoder(self._transforms(input_tensor))

    def encode(self, input_tensor: Tensor) -> Tensor:
        return self._encoder.encode(self._transforms(input_tensor))

    def move_to_device(self, device: DeviceLike):
        self.current_device = torch.device(device)
        self._encoder.to(device)

    def save_checkpoint(self, file_path: Path):
        current_device = self.current_device

        self.move_to_device("cpu")
        torch.save(self._encoder.state_dict(), file_path)
        self.move_to_device(current_device)

    def load_checkpoint(self, file_path: Path):
        current_device = self.current_device

        self.move_to_device("cpu")
        self._encoder.load_state_dict(torch.load(file_path))
        self.move_to_device(current_device)

    @no_grad()
    def encode_dataset(self, dataset: Dataset, batch_size: int = 1) -> Tensor:
        self._encoder.eval()

        data_loader = DataLoader(dataset, batch_size=batch_size)

        encoded_tensors: list[Tensor] = []

        for batch_tensor in tqdm(data_loader, desc="Encoding progress"):
            encoded_tensor = self.encode(batch_tensor)
            encoded_tensors.append(encoded_tensor)

        return torch.cat(encoded_tensors)


class SimCLREncoderModel(EncoderModel):
    def __init__(
        self,
        transforms: list[Module] | None = None,
        checkpoint_path: Path | None = None,
    ):
        encoder = SimCLREncoder()

        super().__init__(
            encoder, transforms=transforms, checkpoint_path=checkpoint_path
        )

        # Initialized on first augment call
        self._augment_transforms: Compose | None = None

    def train(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        train_params: SimCLREncoderTrainParams,
    ) -> tuple[list[float], list[float], int]:
        train_data_loader = DataLoader(
            train_dataset, batch_size=train_params["batch_size"], shuffle=True
        )
        test_data_loader = DataLoader(
            test_dataset, batch_size=train_params["batch_size"], shuffle=True
        )

        loss_function = NTXentLoss(train_params["loss_temperature"])
        optimizer = LARS(
            self._encoder.parameters(),
            base_learning_rate=train_params["base_learning_rate"],
            weight_decay=0.00001,  # same as representation learner (don't remember why)
        )

        train_losses: list[float] = []
        test_losses: list[float] = []

        best_test_loss = float("inf")
        best_epoch = -1
        best_model_state_dict = None

        for epoch in range(train_params["epochs"]):
            print(f"Epoch {epoch + 1} / {train_params['epochs']}")
            print("------------------------------------------------------------")

            train_loss = self._train_epoch(
                loss_function,
                optimizer,
                train_data_loader,
                train_params["output_interval"],
            )
            train_losses.append(train_loss)

            print(f"\nMean train loss: {train_loss:.3e}\n")

            test_loss = self._test(test_data_loader, loss_function)
            test_losses.append(test_loss)

            print(f"Mean test loss: {test_loss:.3e}\n")

            if test_loss < best_test_loss:
                print("New best model found. Saving state dict...\n")

                best_test_loss = test_loss
                best_epoch = epoch + 1
                best_model_state_dict = self._encoder.state_dict()

        print("Training finished.")

        if best_model_state_dict is not None:
            print(f"Loading best model state dict from epoch {best_epoch}...")
            self._encoder.load_state_dict(best_model_state_dict)

        return train_losses, test_losses, best_epoch

    def _train_epoch(
        self,
        loss_function: NTXentLoss,
        optimizer: LARS,
        train_data_loader: DataLoader,
        output_interval: int,
    ) -> float:
        self._encoder.train()

        total_loss = 0

        for batch_index, batch_images in enumerate(train_data_loader):
            transformed_images = self._transforms(batch_images.to(self.current_device))
            augmented_images = self._augment_batch_images(transformed_images)

            loss = loss_function(augmented_images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_index % output_interval == 0:
                batch_loss = loss.item()
                batch_progress = (
                    batch_index + 1
                ) * train_data_loader.batch_size  # type: ignore
                batch_total = len(train_data_loader.dataset)  # type: ignore

                print(
                    f"Loss: {batch_loss:.2e}  [{batch_progress:>4} / {batch_total:>4}]"
                )

        mean_loss = total_loss / len(train_data_loader)

        return mean_loss

    @no_grad()
    def _test(self, test_data_loader: DataLoader, loss_function: NTXentLoss) -> float:
        self._encoder.eval()

        total_loss = 0

        for batch_images in test_data_loader:
            transformed_images = self._transforms(batch_images.to(self.current_device))
            augmented_images = self._augment_batch_images(transformed_images)

            loss = loss_function(augmented_images)

            total_loss += loss.item()

        mean_loss = total_loss / len(test_data_loader)

        return mean_loss

    def _random_augment(self, images_tensor: Tensor) -> Tensor:
        if self._augment_transforms is None:
            image_size = images_tensor.shape[-1]
            blur_kernel_size = (0.2 * image_size) // 2 * 2 + 1

            self._augment_transforms = Compose(
                [
                    RandomCroppedRotation(
                        180, antialias=True, interpolation=InterpolationMode.BILINEAR
                    ),
                    RandomResizedCrop(image_size, scale=(0.5, 1), antialias=True),
                    RandomHorizontalFlip(),
                    ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
                    RandomApply([GaussianBlur(blur_kernel_size)]),
                ]
            )

        return self._augment_transforms(images_tensor)

    def _augment_batch_images(self, batch_images: Tensor) -> Tensor:
        with torch.no_grad():
            augmented_data_1 = self._random_augment(batch_images)
            augmented_data_2 = self._random_augment(batch_images)

            new_batch_size = batch_images.shape[0] * 2

            augment_indexes_1 = torch.arange(0, new_batch_size, 2)
            augment_indexes_2 = augment_indexes_1 + 1

            augmented_data = torch.zeros(new_batch_size, *batch_images.shape[1:]).to(
                batch_images.device
            )

            augmented_data[augment_indexes_1] = augmented_data_1
            augmented_data[augment_indexes_2] = augmented_data_2

        return self._encoder(augmented_data)


class AutoencoderModel(EncoderModel):
    def __init__(
        self,
        image_channels: int = 3,
        image_resolution: int = 224,
        transforms: list[Module] | None = None,
        checkpoint_path: Path | None = None,
    ):
        encoder = Autoencoder(
            image_channels=image_channels, image_resolution=image_resolution
        )

        super().__init__(
            encoder, transforms=transforms, checkpoint_path=checkpoint_path
        )

        self._augment_transforms = Compose(
            [
                RandomCroppedRotation(
                    180, antialias=True, interpolation=InterpolationMode.BILINEAR
                ),
                RandomResizedCrop(image_resolution, scale=(0.5, 1), antialias=True),
                RandomHorizontalFlip(),
            ]
        )

    def train(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        train_params: AutoencoderTrainParams,
    ) -> tuple[list[float], list[float], int]:
        train_data_loader = DataLoader(
            train_dataset, batch_size=train_params["batch_size"], shuffle=True
        )
        test_data_loader = DataLoader(
            test_dataset, batch_size=train_params["batch_size"], shuffle=True
        )

        loss_function = MSELoss()
        optimizer = Adam(self._encoder.parameters(), lr=train_params["learning_rate"])

        train_losses: list[float] = []
        test_losses: list[float] = []

        best_test_loss = float("inf")
        best_epoch = -1
        best_model_state_dict = None

        for epoch in range(train_params["epochs"]):
            print(f"Epoch {epoch + 1} / {train_params['epochs']}")
            print("------------------------------------------------------------")

            train_loss = self._train_epoch(
                loss_function,
                optimizer,
                train_data_loader,
                train_params["output_interval"],
            )
            train_losses.append(train_loss)

            print(f"\nMean train loss: {train_loss:.3e}\n")

            test_loss = self._test(test_data_loader, loss_function)
            test_losses.append(test_loss)

            print(f"Mean test loss: {test_loss:.3e}\n")

            if test_loss < best_test_loss:
                print("New best model found. Saving state dict...\n")

                best_test_loss = test_loss
                best_epoch = epoch + 1
                best_model_state_dict = self._encoder.state_dict()

        print("Training finished.")

        if best_model_state_dict is not None:
            print(f"Loading best model state dict from epoch {best_epoch}...")
            self._encoder.load_state_dict(best_model_state_dict)

        return train_losses, test_losses, best_epoch

    def _train_epoch(
        self,
        loss_function: MSELoss,
        optimizer: Adam,
        train_data_loader: DataLoader,
        output_interval: int,
    ) -> float:
        self._encoder.train()

        total_loss = 0

        for batch_index, batch_images in enumerate(train_data_loader):
            transformed_images = self._transforms(batch_images.to(self.current_device))
            augmented_images = self._augment_transforms(transformed_images)
            reconstructed_images = self._encoder(augmented_images)

            loss = loss_function(reconstructed_images, augmented_images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_index % output_interval == 0:
                batch_loss = loss.item()
                batch_progress = (
                    batch_index + 1
                ) * train_data_loader.batch_size  # type: ignore
                batch_total = len(train_data_loader.dataset)  # type: ignore

                print(
                    f"Loss: {batch_loss:.2e}  [{batch_progress:>4} / {batch_total:>4}]"
                )

        mean_loss = total_loss / len(train_data_loader)

        return mean_loss

    @no_grad()
    def _test(self, test_data_loader: DataLoader, loss_function: MSELoss) -> float:
        self._encoder.eval()

        total_loss = 0

        for batch_images in test_data_loader:
            transformed_images = self._transforms(batch_images.to(self.current_device))
            reconstructed_images = self._encoder(transformed_images)
            loss = loss_function(reconstructed_images, transformed_images)
            total_loss += loss.item()

        mean_loss = total_loss / len(test_data_loader)

        return mean_loss


class SimpleEncoderModel(EncoderModel):
    def __init__(
        self,
        transforms: list[Module] | None = None,
        checkpoint_path: Path | None = None,
    ):
        encoder = SimpleEncoder()

        super().__init__(encoder, transforms, checkpoint_path)
