"""
Module to define the SERes model define in :
Ghosal, P., Nandanwar, L., Kanchan, S., Bhadra, A., Chakraborty, J., & Nandi, D. (2019).
Brain Tumor Classification Using ResNet-101 Based Squeeze and Excitation Deep Neural Network.
in 2019 Second International Conference on Advanced Computational and Communication Paradigms 
(ICACCP) (pp. 1‑6). https://doi.org/10.1109/ICACCP.2019.8882973
"""

from collections.abc import Sequence

import torch
from torch import nn

from src.network.archi import Classifier, Encoder, Model


class SqueezeNExcite(nn.Module):
    """SqueezeNExcite path for SE Res module"""

    def __init__(self, in_channels: int, reduction: int = 2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.module = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the output of the SqueezeNExcite path

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: SqueezeNExcite activation
        """
        batch, channel = x.shape[:2]
        y = self.pool(x).view(batch, channel)
        w = self.module(y).view(batch, channel, 1, 1, 1)
        return x * w


class SEResModule(nn.Module):
    """SERes Module for the SERes Encoder"""

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
    ):
        super().__init__()

        self.key = f"{in_channel}-{out_channel}"
        self.main_path = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, 3, padding="same"),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
            nn.Conv3d(out_channel, out_channel, 3, padding="same"),
            nn.BatchNorm3d(out_channel),
            SqueezeNExcite(out_channel),
            nn.ReLU(),
        )
        self.res_path = nn.Conv3d(in_channel, out_channel, 1, padding="same")
        self.out = nn.Sequential(nn.ReLU(), nn.MaxPool3d(2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute output of one SERes unit

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: SERes module output
        """
        main = self.main_path(x)
        res = self.res_path(x)

        return self.out(main + res)


class SEResEncoder(Encoder):
    """SERes Encoder for the SERes model"""

    _latent_size: int

    def __init__(self, im_shape: Sequence, dropout_rate: float):
        super().__init__(im_shape=im_shape, dropout_rate=dropout_rate)
        self.convs = nn.Sequential(
            SEResModule(1, 8),
            SEResModule(8, 16),
            SEResModule(16, 32),
            SEResModule(32, 64),
            SEResModule(64, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the encoding of the SERes Encoder

        Args:
            x (torch.Tensor): input volume

        Returns:
            torch.Tensor: encoding for the volume
        """
        return self.convs(x)


class SEResClassifier(Classifier):
    """SERes classifier for the SERes model"""

    input_size: int

    def __init__(self, input_size: int, num_classes: int, dropout_rate: float):
        super().__init__(input_size, num_classes, dropout_rate)

        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Flatten(),
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes),
        )


class SEResModel(Model):
    """
    Implementation of the model from Ghosal, P. et al.
    Brain Tumor Classification Using ResNet-101 Based Squeeze and Excitation Deep Neural Network.
    https://doi.org/10.1109/ICACCP.2019.8882973
    """

    def __init__(self, im_shape: Sequence, num_classes: int, dropout_rate: float):
        super().__init__(
            im_shape=im_shape, num_classes=num_classes, dropout_rate=dropout_rate
        )
        self.encoder = SEResEncoder(self.im_shape, self.dropout_rate)
        self.classifier = SEResClassifier(
            self.encoder.latent_size, self.num_classes, self.dropout_rate
        )

    def change_classifier(self, num_classes):
        self.num_classes = num_classes
        self.classifier = SEResClassifier(
            self.encoder.latent_size, self.num_classes, self.dropout_rate
        )
