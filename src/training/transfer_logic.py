"""Module to define lightning logic when using 
transfer learning on pretrained module"""

import torch.optim
from torch import nn
from src.network.transfer_net import TransferMLP
from src.training.common_logic import BaseFinalTrain
from src.network.archi import Model


class TransferTask(BaseFinalTrain):
    """Common class for task to transfer"""

    model: Model
    output_pipeline: nn.Module
    label_loss: nn.Module
    batch_size: int
    output_size: int

    def __init__(self, input_size: int, lr=1e-5, batch_size=14, pool=False):
        super().__init__()
        self.lr = lr
        self.input_size = input_size
        self.model = TransferMLP(self.input_size, self.output_size, pool=pool)
        self.batch_size = batch_size
        self.save_hyperparameters()

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim


class MrArtTransferTask(TransferTask):
    """Task to transfer on MR-ART"""

    output_size = 3
    output_pipeline = nn.Identity()
    label_loss = nn.CrossEntropyLoss()

    def raw_to_pred(self, pred: torch.Tensor) -> torch.Tensor:
        return pred.argmax(dim=1)


class AMPSCZTransferTask(TransferTask):
    """Task to transfer on AMPSCZ"""

    output_size = 1
    output_pipeline = nn.Sequential(
        nn.Flatten(start_dim=0),
    )
    label_loss = nn.BCEWithLogitsLoss()

    def raw_to_pred(self, pred: torch.Tensor) -> torch.Tensor:
        return pred.sigmoid().round().int()
