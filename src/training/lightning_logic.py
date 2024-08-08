"""Define the Lightning Module corresponding to our different tasks:
- Training from scratch
- Pretraining
- Finetuning
for every Dataset"""

import gc
import abc

import lightning
import torch.optim
from torch import nn
from monai.transforms import CutOut
from monai.data.meta_tensor import MetaTensor
from sklearn.metrics import balanced_accuracy_score, r2_score
from src.config import N_BINS
from src.training.callback import get_calibration_curve
from src.transforms.load import ToSoftLabel
from src.network.archi import Model
from src.network.utils import init_weights, parse_model, KLDivLoss


class BaseTrain(abc.ABC, lightning.LightningModule):
    """
    Base class used for training from scratch and finetuning tasks
    """

    model: Model
    output_pipeline: nn.Module
    label_loss: nn.Module
    label: list[float | int] = []
    prediction: list[float | int] = []

    def forward(self, x: torch.Tensor):
        raw_output = self.model(x)
        return self.output_pipeline(raw_output)

    def training_step(self, batch, _):
        volume = batch["data"]
        label = batch["label"]
        prediction = self.forward(volume)
        label_loss = self.label_loss(prediction, label)
        self.log("train_loss", label_loss.item())

        gc.collect()

        return label_loss

    def validation_step(self, batch, _):
        volume = batch["data"]
        label = batch["label"]
        prediction = self.forward(volume)

        label_loss = self.label_loss(prediction, label)
        self.log("val_loss", label_loss.item())

        lab = label.detach().cpu()
        prediction = prediction.detach().cpu()
        self.label += lab.int().tolist()
        self.prediction += self.raw_to_pred(prediction).tolist()
        return label_loss

    def on_validation_epoch_end(self) -> None:
        self.logger.experiment.log_confusion_matrix(
            self.label, self.prediction, epoch=self.current_epoch
        )

        self.log(
            "val_balanced_accuracy",
            balanced_accuracy_score(self.label, self.prediction),
            sync_dist=True,
        )
        self.label = []
        self.prediction = []

    @abc.abstractmethod
    def raw_to_pred(self, pred: torch.Tensor) -> torch.Tensor:
        """Transform raw output of the model to a final prediction

        Args:
            pred (torch.Tensor): Raw prediction

        Returns:
            torch.Tensor: final prediction
        """

    def predict_step(self, batch, _):
        ## VAE TESTING PHASE##
        # INFERENCE
        volume = batch["data"]
        _, prediction = self.forward(volume)

        prediction = self.treat_data_for_pred(prediction)
        return prediction

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.05)
        return optim


class TrainScratchTask(BaseTrain):
    """Common class for task to train from scratch"""

    num_classes: int

    def __init__(
        self,
        model_class: str,
        im_shape,
        lr=1e-5,
        dropout_rate=0.5,
    ):
        super().__init__()
        self.im_shape = im_shape
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.model_class = parse_model(model_class)
        self.model = self.model_class(
            self.im_shape, self.num_classes, self.dropout_rate
        )
        if model_class != "ViT":
            self.model.apply(init_weights)

        self.setup_training()
        self.save_hyperparameters()


class FinetuningTask(BaseTrain):
    """Common class for task to finetune"""

    model: Model
    output_pipeline: nn.Module
    label_loss: nn.Module

    def __init__(
        self,
        pretrained_model: Model,
        im_shape,
        lr=1e-5,
    ):
        super().__init__()
        self.im_shape = im_shape
        self.lr = lr
        self.model = pretrained_model
        self.setup_training()
        self.save_hyperparameters()

    @abc.abstractmethod
    def setup_training(self):
        """This function need to define the label loss and the output pipeline
        for your specific finetuning task"""


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


class MRArtFinetuningTask(FinetuningTask):
    """Task to finetune on MR-ART"""

    def setup_training(self):
        """Function used to define output pipeline and label loss
        and change output num"""
        self.output_pipeline = nn.Identity()
        self.label_loss = nn.CrossEntropyLoss()
        self.model.classifier.change_output_num(3)

    def raw_to_pred(self, pred: torch.Tensor) -> torch.Tensor:
        return pred.argmax(dim=1)


class AMPSCZFinetuningTask(FinetuningTask):
    """Task to finetune on AMPSCZ"""

    def setup_training(self):
        """Function used to define output pipeline and label loss
        and change output num"""
        self.output_pipeline = nn.Sequential(
            nn.Flatten(start_dim=0),
        )
        self.label_loss = nn.BCEWithLogitsLoss()
        self.model.classifier.change_output_num(1)

    def raw_to_pred(self, pred: torch.Tensor) -> torch.Tensor:
        return pred.sigmoid().round().int()


class PretrainingTask(lightning.LightningModule):
    """Pretraining Task in lightning"""

    model: Model
    output_pipeline: nn.Module
    label_loss: nn.Module
    label: list[float | int] = []
    prediction: list[float | int] = []

    def __init__(
        self,
        model_class: str,
        im_shape,
        lr=1e-5,
        dropout_rate=0.5,
        batch_size=14,
        use_cutout=False,
    ):
        super().__init__()
        self.im_shape = im_shape
        self.num_classes = N_BINS
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.lr = lr
        self.model_class = parse_model(model_class)
        self.model = self.model_class(
            self.im_shape, self.num_classes, self.dropout_rate
        )
        if model_class != "VIT":
            self.model.apply(init_weights)

        self.output_pipeline = nn.Sequential(nn.LogSoftmax(dim=1))
        self.label_loss = KLDivLoss()
        self.soft_label_util = ToSoftLabel.baseConfig()

        self.use_cutout = use_cutout
        if self.use_cutout:
            self.cutout = CutOut(self.batch_size)
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor):
        raw_output = self.model(x)
        return self.output_pipeline(raw_output)

    def training_step(self, batch, _):
        volumes: MetaTensor = batch["data"]
        labels = batch["label"]

        if self.use_cutout:
            augvolumes = self.cutout(volumes)
        else:
            augvolumes = volumes

        predictions = self.forward(augvolumes)
        label_loss = self.label_loss(predictions, labels)
        self.log("train_loss", label_loss.item())

        gc.collect()

        return label_loss

    def validation_step(self, batch, _):
        volume = batch["data"]
        label = batch["label"]
        prediction = self.forward(volume)

        label_loss = self.label_loss(prediction, label)
        self.log("val_loss", label_loss.item(), sync_dist=True)

        lab = batch["motion_mm"].detach().cpu()

        self.label += lab.tolist()
        self.prediction += self.soft_label_util.softLabelToHardLabel(
            prediction.detach()
        ).tolist()
        return label_loss

    def on_validation_epoch_end(self) -> None:
        self.log("r2_score", r2_score(self.label, self.prediction), sync_dist=True)
        self.logger.experiment.log_figure(
            figure=get_calibration_curve(self.prediction, self.label),
            figure_name="calibration",
            step=self.current_epoch,
        )
        self.label = []
        self.prediction = []

    def predict_step(self, batch, _):
        ## VAE TESTING PHASE##
        # INFERENCE
        volume = batch["data"]
        prediction = self.forward(volume)
        prediction = self.soft_label_util.softLabelToHardLabel(prediction)

        return prediction

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=0.75, patience=10
        )
        return [optim], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            }
        ]
