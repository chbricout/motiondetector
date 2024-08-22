"""
Module to define the Simple Fully Convolutionnal Network from
Peng, H., Gong, W., Beckmann, C. F., Vedaldi, A., & Smith, S. M. (2021).
Accurate brain age prediction with lightweight deep neural networks.
Medical Image Analysis, 68, 101871. https://doi.org/10.1016/j.media.2020.101871
"""

from collections.abc import Sequence
import torch
from torch import nn
from src.network.archi import Encoder, Classifier, Model


class SFCNBlock(nn.Module):
    """
    SFCN block for the SFCN Encoder
    """

    def __init__(self, kernel_size, in_channel, out_channel, pool=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.BatchNorm3d(out_channel),
        )
        if pool:
            self.block.append(nn.MaxPool3d(2, 2))
        self.block.append(nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output of the simple convolution module

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: output of the convolution module
        """
        return self.block(x)


class SFCNHeadBlock(nn.Sequential):
    """SFCN head bloc, used for classification"""

    def __init__(self, pool_size, in_channel, out_channel, dropout_rate):
        super().__init__(
            nn.AvgPool3d(pool_size),
            nn.Dropout(p=dropout_rate),
            nn.Conv3d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=1,
                padding="same",
            ),
            nn.Flatten(),
        )


class SFCNEncoder(Encoder):
    """SFCN Encoder for SFCN Model"""

    def __init__(self, im_shape: Sequence, dropout_rate: float):
        super().__init__(im_shape=im_shape, dropout_rate=dropout_rate)
        self.convs = nn.Sequential(
            SFCNBlock(3, 1, 32),
            SFCNBlock(3, 32, 64),
            SFCNBlock(3, 64, 128),
            SFCNBlock(3, 128, 256),
            SFCNBlock(3, 256, 256),
            SFCNBlock(1, 256, 64, pool=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the encoding of the SFCN encoder

        Args:
            x (torch.Tensor): input volume tensor

        Returns:
            torch.Tensor: volume's encoding
        """
        return self.convs(x)


class SFCNClassifier(Classifier):
    """SFCN Classifier for the SFCN Model"""

    def __init__(self, input_size: Sequence, num_classes: int, dropout_rate: float):
        super().__init__(input_size, num_classes, dropout_rate)

        self.classifier = SFCNHeadBlock(
            self.input_size, 64, self.num_classes, self.dropout_rate
        )


class SFCNModel(Model):
    """
    Implementation of the model from Han Peng et al. in
    "Accurate brain age prediction with lightweight deep neural networks"
    https://doi.org/10.1016/j.media.2020.101871
    """

    def __init__(self, im_shape: Sequence, num_classes: int, dropout_rate: float):
        super().__init__(
            im_shape=im_shape, num_classes=num_classes, dropout_rate=dropout_rate
        )
        self.encoder = SFCNEncoder(self.im_shape, self.dropout_rate)
        self.classifier = SFCNClassifier(
            self.encoder.latent_shape[1:], self.num_classes, self.dropout_rate
        )
