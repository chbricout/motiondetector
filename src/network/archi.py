import abc
import torch
import torch.nn as nn
from collections.abc import Sequence


class Encoder(abc.ABC, nn.Module):
    """
    Base encoder for the pretraining / finetuning process.
    """

    _latent_shape: Sequence
    _latent_size: int = -1

    im_shape: Sequence
    dropout_rate: float

    def __init__(self, im_shape: Sequence, dropout_rate: float):
        """
        Args:
            im_shape (Sequence): Shape of input data (4D)
            dropout_rate (float): Dropout rate
        """
        super(Encoder, self).__init__()
        self.im_shape = im_shape
        self.dropout_rate = dropout_rate

    def _retrieve_latent(self):
        if self._latent_size<=0:
            shape_like = (1, *self.im_shape)
            out_encoder: torch.Tensor = self.forward(torch.empty(shape_like))
            self._latent_shape = out_encoder.shape[2:]
            self._latent_size = out_encoder.numel()


    @property
    def latent_shape(self) -> Sequence  :
        """Procedure to compute and store the latent size of the model"""
        self._retrieve_latent()
        return self._latent_shape

    @property
    def latent_size(self) -> int:
        """Procedure to compute and store the latent size of the model"""
        self._retrieve_latent()
        return self._latent_size


class Classifier(abc.ABC, nn.Module):
    """
    Base classifier class for the pretraining / finetuning process
    This module should not contain final activation, it has to be handled by the lightning module for more flexibility
    """

    input_size: Sequence | int
    num_classes: int
    dropout_rate: float
    output_layer: nn.Module
    classifier: nn.Module

    @abc.abstractmethod
    def __init__(
        self, input_size: Sequence | int, num_classes: int, dropout_rate: float
    ):
        """
        Args:
            im_shape (Sequence | int): Shape of input data ()
            dropout_rate (float): _description_
        """
        super(Classifier, self).__init__()
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

    def class_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward everything through classifier main block (excluding output layer)

        Args:
            x (torch.Tensor): Input volumes encodings (2D)

        Returns:
            torch.Tensor: Output before last layer
        """
        return self.classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward mechanism, ideally should not be override

        Args:
            x (torch.Tensor):  Input volumes encodings (2D)

        Returns:
            torch.Tensor: raw output
        """
        x = self.class_forward(x)
        return self.output_layer(x)

    @abc.abstractmethod
    def change_output_num(self, num_classes: int):
        pass


class Model(abc.ABC, nn.Module):
    encoder: Encoder
    classifier: Classifier
    im_shape: Sequence
    dropout_rate: float
    num_classes: int

    @abc.abstractmethod
    def __init__(self, im_shape: Sequence, num_classes: int, dropout_rate: float):
        """
        Args:
            im_shape (Sequence): Shape of input data (4D)
            num_classes (int): Number of output classes for classifier
            dropout_rate (float): Dropout rate
        """
        super(Model, self).__init__()
        self.im_shape = im_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        pass

    def encode_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass the batched 3D volumes through the encoder and output only the relevant encoding to feed to the classifier

        Args:
            x (torch.Tensor): Input data (3D volumes)

        Returns:
            torch.Tensor: encoding to feed to a classifier
        """
        return self.encoder(x)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """Default forward mechanism, ideally should not be override

        Args:
            x (torch.Tensor): Input data (3D volumes)
        """
        encoded = self.encoder(x)
        return self.classifier(encoded)

    def mc_dropout(self):
        """Function to enable the dropout layers during test-time"""
        self.eval()
        for m in self.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()
