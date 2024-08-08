"""Module use to define a model compliant to our framework from monai ViT"""

from collections.abc import Sequence
import torch
from torch import nn
from monai.networks.nets import vit
from src.network.archi import Classifier, Encoder, Model


class ViTEncoder(Encoder):
    """ViT Encoder for the ViT model"""

    def __init__(self, im_shape: Sequence, dropout_rate: float):
        super().__init__(im_shape=im_shape, dropout_rate=dropout_rate)
        self.vit = vit.ViT(
            1,
            (160, 192, 160),
            patch_size=(14, 14, 14),
            num_layers=12,
            hidden_size=768,
            classification=True,
            num_classes=1,
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
        )
        self.output_layer = nn.Linear(self.input_size, self.num_classes)

    def change_output_num(self, num_classes: int):
        """Change the size of output layer

        Args:
            num_classes (int): Number of class / length of new output layer
        """
        self.num_classes = num_classes
        self.output_layer = nn.Linear(self.input_size, self.num_classes)


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
