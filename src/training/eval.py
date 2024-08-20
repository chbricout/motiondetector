"""Module defining the callback used at on the Pretraining and Finetuning task
 and any needed function"""

import logging
from typing import Sequence
import comet_ml
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from monai.data.dataloader import DataLoader
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
import pandas as pd
import seaborn as sb
import torch
from src.dataset.ampscz.ampscz_dataset import FinetuneValAMPSCZ
from src.dataset.mrart.mrart_dataset import TrainMrArt, ValMrArt
from src.training.lightning_logic import PretrainingTask
from src.utils.comet import log_figure_comet
from src.transforms.load import FinetuneTransform
from src.utils.metrics import separation_capacity


def get_correlations(model: PretrainingTask, exp: comet_ml.BaseExperiment):
    """Plot and store prediction of a pretrain model on a finetuning task (MR-ART and AMPSCZ)

    Args:
        model (nn.Module): Pretrained model to test.
        exp (comet_ml.BaseExperiment): Experiment to log on.
    """
    load_tsf = FinetuneTransform()
    for dataset in ((TrainMrArt, ValMrArt), FinetuneValAMPSCZ):
        dl = DataLoader(dataset.from_env(load_tsf))
        res = get_pred_from_pretrain(model, dl)
        acc, fig_thresh, thresholds = separation_capacity(res["label"], res["pred"])
        exp.log_metric(f"{dataset.__name__}-acc", acc)
        exp.log_table(f"{dataset.__name__}-pred.csv", res)
        exp.log_other(f"{dataset.__name__}-thresholds-value", thresholds)
        fig_box = get_box_plot(res["pred"], res["label"])
        log_figure_comet(fig_box, f"{dataset.__name__}-calibration", exp=exp)
        log_figure_comet(fig_thresh, f"{dataset.__name__}-thresholds", exp=exp)


def get_pred_from_pretrain(
    model: PretrainingTask, dataloader: DataLoader
) -> pd.DataFrame:
    """Compute prediction of a model on a dataloader

    Args:
        model (nn.Module): Model to use for prediction
        dataloader (DataLoader): Dictionnary based dataloader

    Returns:
        pd.DataFrame: results dataframe containing "pred", "identifier" and "label
    """
    model = model.cuda().eval()
    preds = []
    labels = []
    ids = []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch["data"] = batch["data"].cuda()
            prediction = model.predict_step(batch, idx)
            prediction = prediction.cpu()

            preds += prediction.tolist()
            labels += batch["label"].tolist()
            ids += batch["identifier"]
            torch.cuda.empty_cache()

    full = pd.DataFrame(columns=["pred"])
    full["pred"] = preds
    full["identifier"] = ids
    full["label"] = labels
    return full


def get_box_plot(
    prediction: Sequence[int | float], label: Sequence[int | float]
) -> Figure:
    """Generate box plot of model predictions against ground truth label to visualize distribution

    Args:
        prediction (Sequence[int | float]): prediction vector
        label (Sequence[int | float]): ground truth vector

    Returns:
        Figure: matplotlib's Figure object for the plot
    """
    fig = plt.figure(figsize=(6, 5))
    sb.boxplot(x=label, y=prediction)
    plt.xlabel("Correct Label")
    plt.ylabel("Estimated Motion")
    return fig


class SaveBestCheckpoint(ModelCheckpoint):
    """Callback for the Pretraining process.
    Inherits from ModelCheckpoint to access the best model"""

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule):
        """On fit function end, log best model checkpoint,

        Args:
            trainer (Trainer):  Trainer used for the fit process
            pl_module (LightningModule): Trained Lightning module
            task (str): Pretraining task
        """
        logging.info("Logging pretrain model")
        comet_logger = pl_module.logger
        comet_logger.experiment.log_model(
            name=pl_module.model.__class__.__name__, file_or_folder=self.best_model_path
        )
