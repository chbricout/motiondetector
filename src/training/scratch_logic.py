"""Module to define lightning logic when training from scratch"""

import torch
from torch import nn
from src.network.utils import init_model, parse_model
from src.training.common_logic import BaseFinalTrain


class TrainScratchTask(BaseFinalTrain):
    """Common class for task to train from scratch"""

    num_classes: int
    batch_size: int

    def __init__(
        self, model_class: str, im_shape, lr=1e-5, dropout_rate=0.5, batch_size=14
    ):
        super().__init__()
        self.im_shape = im_shape
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.batch_size = batch_size
        self.model_class = parse_model(model_class)
        self.model = self.model_class(
            self.im_shape, self.num_classes, self.dropout_rate
        )
        init_model(self.model)
        self.setup_training()
        self.save_hyperparameters()
        self.model = torch.compile(self.model)


class MRArtScratchTask(TrainScratchTask):
    """Task to train from scratch on MR-ART"""

    num_classes = 3

    def setup_training(self):
        """Function used to define output pipeline and label loss"""
        self.output_pipeline = nn.Identity()
        self.label_loss = nn.CrossEntropyLoss()

    def raw_to_pred(self, pred: torch.Tensor) -> torch.Tensor:
        return pred.argmax(dim=1)


class AMPSCZScratchTask(TrainScratchTask):
    """Task to train from scratch on AMPSCZ"""

    num_classes = 1

    def setup_training(self):
        """Function used to define output pipeline and label loss"""
        self.output_pipeline = nn.Sequential(
            nn.Flatten(start_dim=0),
        )
        self.label_loss = nn.BCEWithLogitsLoss()

    def raw_to_pred(self, pred: torch.Tensor) -> torch.Tensor:
        return pred.sigmoid().round().int()
