"""Module to define common lightning logic elements"""

from collections.abc import Sequence
import gc
import abc
import lightning
from sklearn.metrics import balanced_accuracy_score
import torch.optim
from torch import nn
from monai.data.meta_tensor import MetaTensor
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sb
from src.network.archi import Model


def get_calibration_curve(
    prediction: Sequence[int | float],
    label: Sequence[int | float],
    hue: Sequence[int] | None = None,
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
    batch_size: int

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
        self.log("train_loss", label_loss.item(), batch_size=self.batch_size)

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
    batch_size: int

    label: list[float | int] = []
    prediction: list[float | int] = []

    def validation_step(self, batch, _):
        volume = batch["data"]
        label = batch["label"]
        prediction = self.forward(volume)

        label_loss = self.label_loss(prediction, label)
        self.log("val_loss", label_loss.item(), batch_size=self.batch_size)

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
            batch_size=self.batch_size,
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
