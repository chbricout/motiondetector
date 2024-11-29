"""Module to define lightning logic when using 
transfer learning on pretrained module"""

import torch.optim
from torch import Tensor, nn

from src.network.archi import Encoder, Model
from src.network.transfer_net import TransferMLP
from src.network.utils import init_model
from src.training.common_logic import BaseFinalTrain


class TransferTask(BaseFinalTrain):
    def __init__(
        self,
        input_size: int,
        pretrained: Model = None,
        encoder: Encoder = None,
        lr=1e-5,
        dropout_rate=0.6,
        batch_size=14,
        weight_decay=0.05,
        num_layers=2,
        pool=False,
    ):
        super().__init__()
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.input_size = input_size
        self.weight_decay = weight_decay
        self.num_layers = num_layers
        if pretrained is not None:
            self.encoder = pretrained.encoder
        elif encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = nn.Identity()
        for weigth in self.encoder.parameters():
            weigth.requires_grad = False
        self.model = TransferMLP(
            self.input_size,
            self.num_classes,
            pool=pool,
            dropout_rate=self.dropout_rate,
            num_layers=self.num_layers,
        )
        init_model(self.model)
        self.batch_size = batch_size

        self.save_hyperparameters()

    def train_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Used for transfer learning on encoding only"""
        return self.model(x)

    def classify(self, embedding: Tensor) -> Tensor:
        classes = self.model(embedding)
        raw = self.output_pipeline(classes)
        return self.raw_to_pred(raw)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoding = self.encoder(x)
        raw_output = self.model(encoding)
        return self.output_pipeline(raw_output)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=0.2, patience=10
        )
        return [optim], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            }
        ]


class AMPSCZTransferTask(TransferTask):
    """Task to transfer on AMPSCZ"""

    num_classes = 3
    output_pipeline = nn.Identity()
    label_loss = nn.CrossEntropyLoss()

    def raw_to_pred(self, pred: torch.Tensor) -> torch.Tensor:
        return pred.argmax(dim=1)
