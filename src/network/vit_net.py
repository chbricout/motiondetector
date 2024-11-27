"""Module use to define a model compliant to our framework from monai ViT"""

from collections.abc import Sequence

import torch
from monai.networks.nets import vit
from torch import nn

from src.network.archi import Classifier, Encoder, Model


class ViTEncoder(Encoder):
    """ViT Encoder for the ViT model"""

    def __init__(self, im_shape: Sequence, dropout_rate: float):
        super().__init__(im_shape=im_shape, dropout_rate=dropout_rate)
        self.vit = vit.ViT(
            1,
            (160, 192, 160),
            patch_size=(16, 16, 16),
            num_layers=12,
            hidden_size=768,
            classification=True,
            num_classes=1,
            dropout_rate=0.1,
        )
        delattr(self.vit, "classification_head")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the ViT encoding (classification token)

        Args:
            x (torch.Tensor): Volume tensor

        Returns:
            torch.Tensor: ViT's encoding (classification token embedding)
        """
        z, _ = self.vit(x)
        return z[:, 0]


class ViTClassifier(Classifier):
    """Classifier for the ViT Model"""

    input_size: int

    def __init__(self, input_size: int, num_classes: int, dropout_rate: float):
        super().__init__(input_size, num_classes, dropout_rate)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.input_size, self.num_classes),
        )


class ViTModel(Model):
    """ViT Model combining the ViT Encoder and Classifier"""

    def __init__(self, im_shape: Sequence, num_classes: int, dropout_rate: float):
        super().__init__(
            im_shape=im_shape, num_classes=num_classes, dropout_rate=dropout_rate
        )
        self.encoder = ViTEncoder(self.im_shape, self.dropout_rate)
        self.classifier = ViTClassifier(
            self.encoder.latent_size, self.num_classes, self.dropout_rate
        )

    def change_classifier(self, num_classes):
        self.num_classes = num_classes
        self.classifier = ViTClassifier(
            self.encoder.latent_size, self.num_classes, self.dropout_rate
        )
