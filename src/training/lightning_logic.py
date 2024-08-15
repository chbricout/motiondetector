"""Define the Lightning Module corresponding to our different tasks:
- Training from scratch
- Pretraining
- Finetuning
for every Dataset"""

from collections.abc import Sequence
import gc
import abc
import logging
import lightning
import torch.optim
from torch import nn
from monai.transforms import CutOut
from monai.data.meta_tensor import MetaTensor
from sklearn.metrics import balanced_accuracy_score, r2_score
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sb
from src import config
from src.transforms.load import ToSoftLabel
from src.network.archi import Model
from src.network.utils import init_model, parse_model, KLDivLoss


def get_calibration_curve(
    prediction: Sequence[int | float],
    label: Sequence[int | float],
    hue: Sequence[int] = None,
) -> Figure:
    """Generate calibration curve with matplotlib's pyplot

    Args:
        prediction (Sequence[int | float]): prediction vector
        label (Sequence[int | float]): ground truth vector
        hue (Sequence[int], optional): vector for hue purpose. Defaults to None.

    Returns:
        Figure: matplotlib's Figure object for the plot
    """
    fig = plt.figure(figsize=(6, 5))
    sb.scatterplot(x=label, y=prediction, hue=hue)
    min_lab = min(label)
    max_lab = max(label)
    plt.plot([min_lab, max_lab], [min_lab, max_lab], "r")
    plt.xlabel("Correct Label")
    plt.ylabel("Estimated Label")
    return fig


class EncodeClassifyTask(abc.ABC, lightning.LightningModule):
    """Basic lightning task describing model that encode and classify
    Unlike `Model`s class which are more generique implementation,
    classes that inherit from this one also implements the post
    processing pipeline needed to get clean results.
    This class is mostly used to have a common strategy of
    optimising mc-dropout
    """

    output_pipeline: nn.Module
    model: Model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_output = self.model(x)
        return self.output_pipeline(raw_output)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input tensor

        Args:
            x (torch.Tensor): volume to encode

        Returns:
            torch.Tensor: resulting embedding
        """
        return self.model.encode_forward(x)

    def classify(self, embedding: torch.Tensor) -> torch.Tensor:
        """Classify an embedding

        Args:
            embedding (torch.Tensor): Embedding to classify

        Returns:
            torch.Tensor: Fully processed class
        """
        raw = self.model.classifier(embedding)
        out = self.output_pipeline(raw)
        return self.raw_to_pred(out)

    def training_step(self, batch, _):
        volumes: MetaTensor = batch["data"]
        labels = batch["label"]

        predictions = self.forward(volumes)

        label_loss = self.label_loss(predictions, labels)
        self.log("train_loss", label_loss.item(), batch_size=self.trainer.datamodule.batch_size)

        gc.collect()

        return label_loss

    @abc.abstractmethod
    def raw_to_pred(self, pred: torch.Tensor) -> torch.Tensor:
        """Transform raw output of the model to a final prediction

        Args:
            pred (torch.Tensor): Raw prediction

        Returns:
            torch.Tensor: final prediction
        """


class BaseFinalTrain(EncodeClassifyTask):
    """
    Base class used for training from scratch and finetuning tasks
    """

    model: Model
    output_pipeline: nn.Module
    label_loss: nn.Module
    label: list[float | int] = []
    prediction: list[float | int] = []

    def validation_step(self, batch, _):
        volume = batch["data"]
        label = batch["label"]
        prediction = self.forward(volume)

        label_loss = self.label_loss(prediction, label)
        self.log("val_loss", label_loss.item(), batch_size=self.trainer.datamodule.batch_size)

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
            batch_size=self.trainer.datamodule.batch_size,
        )
        self.label = []
        self.prediction = []

    def predict_step(self, batch, _):
        volume = batch["data"]
        prediction = self.forward(volume)

        prediction = self.raw_to_pred(prediction)
        return prediction

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.05)
        return optim


class TrainScratchTask(BaseFinalTrain):
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
        init_model(self.model)

        self.setup_training()
        self.save_hyperparameters()


class FinetuningTask(BaseFinalTrain):
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
        model_class: str,
        im_shape,
        lr=1e-5,
        dropout_rate=0.5,
        batch_size=14,
        use_cutout=False,
        num_classes: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.im_shape = im_shape
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.lr = lr
        self.model_class = parse_model(model_class)
        self.model = torch.compile(
            self.model_class(self.im_shape, self.num_classes, self.dropout_rate)
        )
        init_model(self.model)

        self.use_cutout = use_cutout
        if self.use_cutout:
            self.cutout = CutOut(self.batch_size)
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
            batch_size=self.trainer.datamodule.batch_size,
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
            batch_size=self.trainer.datamodule.batch_size,
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
            optim, mode="min", factor=0.75, patience=10
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
        use_cutout=False,
    ):
        super().__init__(
            model_class=model_class,
            im_shape=im_shape,
            lr=lr,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            use_cutout=use_cutout,
            num_classes=config.MOTION_N_BINS,
        )

    def raw_to_pred(self, out: torch.Tensor) -> torch.Tensor:
        return self.soft_label_util.logsoft_to_hardlabel(out)


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
        use_cutout=False,
    ):
        super().__init__(
            model_class=model_class,
            im_shape=im_shape,
            lr=lr,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            use_cutout=use_cutout,
            num_classes=config.SSIM_N_BINS,
        )

    def raw_to_pred(self, out: torch.Tensor) -> torch.Tensor:
        return self.soft_label_util.logsoft_to_hardlabel(out)


class BinaryPretrainingTask(PretrainingTask):
    """
    Pretraining Task for Binary motion prediction task
    """

    hard_label_tag = "motion_binary"
    output_pipeline = nn.Sequential(nn.Sigmoid(), nn.Flatten(start_dim=0))
    label_loss = nn.BCELoss()

    def __init__(
        self,
        model_class: str,
        im_shape,
        lr=1e-5,
        dropout_rate=0.5,
        batch_size=14,
        use_cutout=False,
    ):
        super().__init__(
            model_class=model_class,
            im_shape=im_shape,
            lr=lr,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            use_cutout=use_cutout,
            num_classes=1,
        )

    def raw_to_pred(self, out: torch.Tensor) -> torch.Tensor:
        return out.round().int().flatten()
