"""
Module to define a basic convolution network
"""

from collections.abc import Sequence
import torch
from torch import nn
from monai.networks.blocks import Convolution
from src.network.archi import Model, Encoder, Classifier


class ConvModule(nn.Module):
    """
    Base module for CNN Encoder
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        stride,
        conv_kernel=3,
        act="RELU",
    ):
        super().__init__()
        self.key = f"{in_channel}-{out_channel}"
        padding = conv_kernel // 2
        self.conv_in = Convolution(
            spatial_dims=3,
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=conv_kernel,
            padding=padding,
            strides=stride,
            norm="BATCH",
            act=act,
        )
        self.conv_mid = Convolution(
            spatial_dims=3,
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=conv_kernel,
            strides=1,
            padding=padding,
            norm="BATCH",
            act=act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute one layer of two convolutions for CNN encoder

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: convolutions output
        """
        y = self.conv_in(x)
        y = self.conv_mid(y)

        return y


class CNNEncoder(Encoder):
    """
    Encoder for CNN Model
    """

    def __init__(self, im_shape: Sequence, dropout_rate: float):
        super().__init__(im_shape=im_shape, dropout_rate=dropout_rate)
        self.convs = nn.Sequential(
            ConvModule(1, 32, 2),
            ConvModule(32, 64, 2),
            ConvModule(64, 128, 2),
            ConvModule(128, 256, 2),
            ConvModule(256, 512, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute volume encoding through 5 ConvModule

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: convolutions output
        """
        return self.convs(x)


class CNNClassifier(Classifier):
    """
    Classifier for CNN Model
    """

    input_size: int

    def __init__(self, input_size: int, num_classes: int, dropout_rate: float):
        super().__init__(input_size, num_classes, dropout_rate)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.input_size, self.num_classes),
        )


class CNNModel(Model):
    """
    Combine a CNN encoder and classifier
    """

    def __init__(self, im_shape: Sequence, num_classes: int, dropout_rate: float):
        super().__init__(
            im_shape=im_shape, num_classes=num_classes, dropout_rate=dropout_rate
        )
        self.encoder = CNNEncoder(self.im_shape, self.dropout_rate)
        self.classifier = CNNClassifier(
            self.encoder.latent_size, self.num_classes, self.dropout_rate
        )
