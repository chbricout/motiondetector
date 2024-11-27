"""
Module to define a special kind of Model use for transfer learning
"""

import itertools
from collections.abc import Sequence

from torch import nn

from src.network.archi import Classifier, Encoder, Model


class PoolEncoder(Encoder):
    """Encoder used as a placeholder to have
    our TransferMLP compatible with Model interfaces"""

    def __init__(self, im_shape: Sequence, dropout_rate: float):
        super().__init__(im_shape=im_shape, dropout_rate=dropout_rate)
        self.pool = nn.AvgPool3d(im_shape[1:])

    def forward(self, x):
        """Identity forward function

        Args:
            x (torch.Tensor): input data

        Returns:
            torch.Tensor: exact input data
        """
        res = self.pool(x).flatten(start_dim=1)
        return res


class FlattenEncoder(Encoder):
    """Encoder used as a placeholder to have
    our TransferMLP compatible with Model interfaces"""

    def forward(self, x):
        """Identity forward function

        Args:
            x (torch.Tensor): input data

        Returns:
            torch.Tensor: exact input data
        """
        res = x.flatten(start_dim=1)
        return res


class MLPClassifier(Classifier):
    """A simple MLP network"""

    input_size: int
    num_classes: int
    dropout_rate: float
    classifier: nn.Module

    def __init__(
        self, input_size: int, num_classes: int, dropout_rate: float, num_layers: int
    ):
        super().__init__(
            input_size=input_size, num_classes=num_classes, dropout_rate=dropout_rate
        )
        layers = [self.input_size]
        for _ in range(num_layers):
            layers.append(10)
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            # nn.Linear(self.input_size, 10),
            # nn.BatchNorm1d(10),
            # nn.ReLU(),
            # nn.Linear(10, 10),
            # nn.BatchNorm1d(10),
            # nn.ReLU(),
            # nn.Linear(10, 10),
            # nn.BatchNorm1d(10),
            # nn.ReLU(),
            # nn.Linear(10, self.num_classes),
            # nn.Linear(self.input_size, self.num_classes),
        )
        for i in range(len(layers) - 2 + 1):
            self.classifier.append(nn.Linear(layers[i], layers[i + 1]))
            self.classifier.append(nn.BatchNorm1d(layers[i + 1]))
            self.classifier.append(nn.ReLU())
        self.classifier.append(nn.Linear(layers[-1], self.num_classes))


class TransferMLP(Model):
    """
    Class to implement transfer learning by training a new MLP out of pretrained embeddings.
    Inherits from model to be compatible with every interfaces used
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout_rate: float = 0.7,
        pool=False,
        num_layers=3,
    ):
        super().__init__(
            im_shape=input_size, num_classes=output_size, dropout_rate=dropout_rate
        )
        self.encoder = (
            FlattenEncoder(im_shape=self.im_shape, dropout_rate=self.dropout_rate)
            if not pool
            else PoolEncoder(im_shape=self.im_shape, dropout_rate=self.dropout_rate)
        )
        self.classifier = MLPClassifier(
            self.encoder.latent_size,
            self.num_classes,
            self.dropout_rate,
            num_layers=num_layers,
        )
