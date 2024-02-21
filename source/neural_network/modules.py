import torch
from torch import Tensor
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    Identity,
    MaxPool2d,
    MaxUnpool2d,
    Module,
    ReLU,
    Sequential,
    Unflatten,
    Upsample,
)
from torchvision.models import ResNet18_Weights, resnet18


class DecoderResidualBlock(Module):
    def __init__(
        self, in_channels: int, out_channels: int, upsample: bool = False
    ) -> None:
        super().__init__()

        if upsample:
            self._upsample = Sequential(
                ConvTranspose2d(
                    in_channels, out_channels, 1, stride=2, output_padding=1, bias=False
                ),
                BatchNorm2d(out_channels),
            )
        else:
            self._upsample = Identity()

        conv1_stride = 1 + upsample
        conv1_output_padding = 0 + upsample

        self._relu = ReLU(inplace=True)

        self._conv1 = ConvTranspose2d(
            in_channels,
            out_channels,
            3,
            stride=conv1_stride,
            padding=1,
            output_padding=conv1_output_padding,
            bias=False,
        )
        self._bn1 = BatchNorm2d(out_channels)

        self._conv2 = ConvTranspose2d(
            out_channels, out_channels, 3, padding=1, bias=False
        )
        self._bn2 = BatchNorm2d(out_channels)

    def forward(self, input_tensor: Tensor) -> Tensor:
        identity = self._upsample(input_tensor)

        output_tensor = self._conv1(input_tensor)
        output_tensor = self._bn1(output_tensor)
        output_tensor = self._relu(output_tensor)

        output_tensor = self._conv2(output_tensor)
        output_tensor = self._bn2(output_tensor)

        output_tensor += identity
        output_tensor = self._relu(output_tensor)

        return output_tensor


class DecoderResidualLayer(Module):
    def __init__(
        self, in_channels: int, out_channels: int, upsample: bool = False
    ) -> None:
        super().__init__()

        self._block1 = DecoderResidualBlock(
            in_channels, out_channels, upsample=upsample
        )
        self._block2 = DecoderResidualBlock(out_channels, out_channels)

    def forward(self, input_tensor: Tensor) -> Tensor:
        output_tensor = self._block1(input_tensor)
        output_tensor = self._block2(output_tensor)

        return output_tensor


class DecoderResNet18(Module):
    def __init__(self, *, output_resolution: int, output_channels: int) -> None:
        super().__init__()

        self._inverted_avgpool_size = output_resolution // 32
        self._maxunpool_output_size = (
            -1,
            64,
            output_resolution // 2,
            output_resolution // 2,
        )

        # Part 1
        # self._p1_linear = Linear(1000, 512) class layer removed
        self._p1_unflatten = Unflatten(1, (512, 1, 1))
        self._p1_inverted_avgpool = Upsample(size=self._inverted_avgpool_size)

        # Part 2
        self._p2_layer1 = DecoderResidualLayer(512, 512)
        self._p2_layer2 = DecoderResidualLayer(512, 256, upsample=True)
        self._p2_layer3 = DecoderResidualLayer(256, 128, upsample=True)
        self._p2_layer4 = DecoderResidualLayer(128, 64, upsample=True)

        # Part 3
        self._p3_maxunpool = MaxUnpool2d(3, stride=2, padding=1)
        self._p3_relu = ReLU(inplace=True)  # before or after conv?
        self._p3_conv = ConvTranspose2d(
            64, output_channels, 7, stride=2, padding=3, output_padding=1, bias=False
        )
        self._p3_bn = BatchNorm2d(  # before or after conv?, note bias=False
            output_channels
        )

    def forward(self, input_tensor: Tensor, maxpool_indices: Tensor) -> Tensor:
        # Part 1
        # x = self._p1_linear(input_tensor) class layer removed
        x = self._p1_unflatten(input_tensor)
        x = self._p1_inverted_avgpool(x)

        # Part 2
        x = self._p2_layer1(x)
        x = self._p2_layer2(x)
        x = self._p2_layer3(x)
        x = self._p2_layer4(x)

        # Part 3
        x = self._p3_maxunpool(
            x, maxpool_indices, output_size=self._maxunpool_output_size
        )
        x = self._p3_relu(x)
        x = self._p3_conv(x)
        x = self._p3_bn(x)

        return x


class EncoderResNet18(Module):
    def __init__(self, *, image_channels: int) -> None:
        super().__init__()

        self._resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        if image_channels != 3:
            self._resnet.conv1 = Conv2d(
                image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self._resnet.maxpool = MaxPool2d(
            kernel_size=3, stride=2, padding=1, return_indices=True
        )

    def forward(
        self, input_tensor: Tensor, return_maxpool_indices: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        x = self._resnet.conv1(input_tensor)
        x = self._resnet.bn1(x)
        x = self._resnet.relu(x)
        x, maxpool_indices = self._resnet.maxpool(x)

        x = self._resnet.layer1(x)
        x = self._resnet.layer2(x)
        x = self._resnet.layer3(x)
        x = self._resnet.layer4(x)

        x = self._resnet.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self._resnet.fc(x) class layer removed

        if return_maxpool_indices:
            return x, maxpool_indices

        return x
