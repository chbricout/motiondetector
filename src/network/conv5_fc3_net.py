"""
Module to define the Conv5_FC3 network defined in:
Bottani, S., Burgos, N., Maire, A., Wild, A., Ströer, S., Dormont, D., & Colliot, O. (2022).
Automatic quality control of brain T1-weighted magnetic resonance images for a clinical
data warehouse. Medical Image Analysis, 75, 102219. https://doi.org/10.1016/j.media.2021.102219
"""

from collections.abc import Sequence
import torch
from torch import nn
from src.network.archi import Classifier, Model, Encoder


class ConvBlock(nn.Sequential):
    """Base block for the Conv5_FC3 model"""

    def __init__(self, in_channel, out_channel):
        super().__init__(
            nn.Conv3d(in_channel, out_channel, 3, padding="same"),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
        )


class Conv5FC3Encoder(Encoder):
    """Encoder for Conv5_FC3 model"""

    _latent_size: int

    def __init__(self, im_shape: Sequence, dropout_rate: float):
        super().__init__(im_shape=im_shape, dropout_rate=dropout_rate)
        self.convs = nn.Sequential(
            ConvBlock(1, 8),
            ConvBlock(8, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the encoding for Conv5_FC3 encoder

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: 5 convolution's output
        """
        return self.convs(x)


class Conv5FC3Classifier(Classifier):
    """
    Classifier for Conv5_FC3 model
    """

    input_size: int

    def __init__(self, input_size: int, num_classes: int, dropout_rate: float):
        super().__init__(input_size, num_classes, dropout_rate)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.input_size, 1300),
            nn.Linear(1300, 50),
            nn.Linear(50, self.num_classes),
        )


class Conv5FC3Model(Model):
    """
    Implementation of the model from Simona Bottani et al. in
    "Automatic quality control of brain T1-weighted magnetic resonance images
    for a clinical data warehouse"
    https://doi.org/10.1016/j.media.2021.102219
    """

    def __init__(self, im_shape: Sequence, num_classes: int, dropout_rate: float):
        super().__init__(
            im_shape=im_shape, num_classes=num_classes, dropout_rate=dropout_rate
        )
        self.encoder = Conv5FC3Encoder(self.im_shape, self.dropout_rate)
        self.classifier = Conv5FC3Classifier(
            self.encoder.latent_size, self.num_classes, self.dropout_rate
        )
