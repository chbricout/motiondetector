"""Module to define logic for pretraining"""

import logging
from collections import Counter

import matplotlib.pyplot as plt
import torch.optim
from sklearn.metrics import balanced_accuracy_score, mean_squared_error, r2_score
from torch import nn

from src import config
from src.network.archi import Model
from src.network.utils import KLDivLoss, init_model, parse_model
from src.training.common_logic import EncodeClassifyTask, get_calibration_curve
from src.transforms.load import ToSoftLabel


class PretrainingTask(EncodeClassifyTask):
    """Pretraining Task in lightning"""

    model: Model
    label_loss: nn.Module
    hard_label_tag: str  # Used to retrieve hard label without computation from batch
    label: list[float | int] = []
    prediction: list[float | int] = []
    num_classes: int

    def __init__(
        self,
        model_class: str = "",
        im_shape=config.IM_SHAPE,
        lr=1e-5,
        dropout_rate=0.5,
        batch_size=14,
        num_classes: int = 1,
        pretrained_model: Model = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.im_shape = im_shape
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.lr = lr
        if pretrained_model:
            self.model_class = pretrained_model.__class__
            self.model = pretrained_model
            self.model = torch.compile(pretrained_model)
        else:
            self.model_class = parse_model(model_class)
            self.model = self.model_class(
                self.im_shape, self.num_classes, self.dropout_rate
            )
            init_model(self.model)
            if model_class == "SFCN":
                self.model = torch.compile(self.model, disable=True)
            else:
                self.model = torch.compile(self.model)

        self.save_hyperparameters()

    def validation_step(self, batch, _):
        volume = batch["data"]
        label = batch["label"]
        prediction = self.forward(volume)
        label_loss = self.label_loss(prediction, label)
        self.log(
            "val_loss",
            label_loss.item(),
            sync_dist=True,
            batch_size=self.batch_size,
        )
        lab = batch[self.hard_label_tag].detach().cpu()

        self.label += lab.tolist()
        self.prediction += self.raw_to_pred(prediction.detach()).tolist()
        return label_loss

    def on_validation_epoch_end(self) -> None:
        self.log(
            "r2_score",
            r2_score(self.label, self.prediction),
            sync_dist=True,
            batch_size=self.batch_size,
        )

        self.log(
            "rmse",
            mean_squared_error(self.label, self.prediction, squared=False),
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.logger.experiment.log_figure(
            figure=get_calibration_curve(self.prediction, self.label),
            figure_name="calibration",
            step=self.current_epoch,
        )
        self.label = []
        self.prediction = []
        plt.close()

    def predict_step(self, batch, _):
        volume = batch["data"]
        prediction = self.forward(volume)
        prediction = self.raw_to_pred(prediction)

        return prediction

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=0.6, patience=5
        )
        return [optim], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            }
        ]


class MotionPretrainingTask(PretrainingTask):
    """
    Pretraining Task for Motion mm metric
    """

    hard_label_tag = "motion_mm"
    output_pipeline = nn.LogSoftmax(dim=1)
    label_loss = KLDivLoss()
    soft_label_util: ToSoftLabel = ToSoftLabel.motion_config()

    def __init__(
        self,
        model_class: str,
        im_shape,
        lr=1e-5,
        dropout_rate=0.5,
        batch_size=14,
    ):
        super().__init__(
            model_class=model_class,
            im_shape=im_shape,
            lr=lr,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            num_classes=config.MOTION_N_BINS,
        )

    def raw_to_pred(self, pred: torch.Tensor) -> torch.Tensor:
        return self.soft_label_util.logsoft_to_hardlabel(pred)


class SSIMPretrainingTask(PretrainingTask):
    """
    Pretraining Task for SSIM metric
    """

    hard_label_tag = "ssim_loss"
    output_pipeline = nn.LogSoftmax(dim=1)
    label_loss = KLDivLoss()
    soft_label_util: ToSoftLabel = ToSoftLabel.ssim_config()

    def __init__(
        self,
        model_class: str,
        im_shape,
        lr=1e-5,
        dropout_rate=0.5,
        batch_size=14,
    ):
        super().__init__(
            model_class=model_class,
            im_shape=im_shape,
            lr=lr,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            num_classes=config.SSIM_N_BINS,
        )

    def raw_to_pred(self, pred: torch.Tensor) -> torch.Tensor:
        return self.soft_label_util.logsoft_to_hardlabel(pred)


class BinaryPretrainingTask(PretrainingTask):
    """
    Pretraining Task for Binary motion prediction task
    """

    hard_label_tag = "motion_binary"
    output_pipeline = nn.Sequential(nn.Flatten(start_dim=0))
    label_loss = nn.BCEWithLogitsLoss()

    def __init__(
        self,
        model_class: str,
        im_shape,
        lr=1e-5,
        dropout_rate=0.5,
        batch_size=14,
    ):
        super().__init__(
            model_class=model_class,
            im_shape=im_shape,
            lr=lr,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            num_classes=1,
        )

    def setup(self, stage):
        labels = self.trainer.datamodule.train_dataloader().dataset.file["label"]
        class_counts = Counter(labels)
        total_count = len(labels)
        class_weights = {
            int(cls): 1 - (count / total_count) for cls, count in class_counts.items()
        }
        logging.warn(class_weights)
        self.label_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.as_tensor(class_weights[1])
        )

    def raw_to_pred(self, pred: torch.Tensor) -> torch.Tensor:
        return pred.sigmoid().round().int().flatten()

    def on_validation_epoch_end(self) -> None:
        self.log(
            "balanced_accuracy",
            balanced_accuracy_score(self.label, self.prediction),
            sync_dist=True,
            batch_size=self.batch_size,
        )

        self.log(
            "rmse",
            mean_squared_error(self.label, self.prediction, squared=False),
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.logger.experiment.log_confusion_matrix(
            self.label, self.prediction, epoch=self.current_epoch
        )
        self.label = []
        self.prediction = []

    def predict_step(self, batch, _):
        volume = batch["data"]
        prediction = self.forward(volume)

        return prediction.sigmoid().flatten()


class ContinualPretrainingTask(PretrainingTask):
    """
    Pretraining Task for Binary motion prediction task
    """

    hard_label_tag = "three_motion"
    output_pipeline = nn.Identity()
    label_loss = nn.CrossEntropyLoss()

    def __init__(
        self,
        pretrained: Model,
        lr=1e-5,
        batch_size=14,
    ):
        pretrained.change_classifier(3)
        super().__init__(
            pretrained_model=pretrained,
            lr=lr,
            batch_size=batch_size,
            num_classes=1,
        )

    def raw_to_pred(self, pred: torch.Tensor) -> torch.Tensor:
        return pred.argmax(dim=1).flatten()

    def on_validation_epoch_end(self) -> None:
        self.log(
            "balanced_accuracy",
            balanced_accuracy_score(self.label, self.prediction),
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.logger.experiment.log_confusion_matrix(
            self.label, self.prediction, epoch=self.current_epoch
        )
        self.label = []
        self.prediction = []

    def predict_step(self, batch, _):
        volume = batch["data"]
        prediction = self.forward(volume)
        return prediction.argmax(dim=1).flatten()


class ContinualMotionPretrainingTask(ContinualPretrainingTask):
    def __init__(self, pretrained: Model, lr=1e-5, batch_size=14):
        super().__init__(pretrained, lr, batch_size)


class ContinualSSIMPretrainingTask(ContinualPretrainingTask):
    def __init__(
        self,
        pretrained: Model,
        lr=1e-5,
        batch_size=14,
    ):
        super().__init__(
            pretrained,
            lr,
            batch_size,
        )
