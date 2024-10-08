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
from sklearn.metrics import balanced_accuracy_score
import torch
import tqdm
from src.dataset.ampscz.ampscz_dataset import TransferValAMPSCZ, TransferTrainAMPSCZ
from src.dataset.mrart.mrart_dataset import TrainMrArt, ValMrArt
from src.training.pretrain_logic import PretrainingTask
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
    task_conf = {
        "MRART": {"train": TrainMrArt, "val": ValMrArt},
        "AMPSCZ": {"train": TransferTrainAMPSCZ, "val": TransferValAMPSCZ},
    }
    for data_name, datasets_mode in task_conf.items():
        all_mode_df = []
        for mode, dataset in datasets_mode.items():
            dl = DataLoader(dataset.from_env(load_tsf))
            all_mode_df.append(get_pred_from_pretrain(model, dl, mode))
        all_mode_df = pd.concat(all_mode_df)
        acc, per_class_accuracy, thresholds,fig_thresh, _ = separation_capacity(all_mode_df)
        exp.log_metric(f"{data_name}-acc", acc)
        exp.log_table(f"{data_name}-pred.csv", all_mode_df)
        exp.log_other(f"{data_name}-thresholds-value", thresholds)
        exp.log_other(f"{data_name}-per-class-accuracy", per_class_accuracy)

        fig_box = get_box_plot(all_mode_df["pred"], all_mode_df["label"])
        log_figure_comet(fig_box, f"{data_name}-calibration", exp=exp)
        log_figure_comet(fig_thresh, f"{data_name}-thresholds", exp=exp)


def get_pred_from_pretrain(
    model: PretrainingTask,
    dataloader: DataLoader,
    mode: str = "test",
    label: str = "label",
    cuda=True
) -> pd.DataFrame:
    """Compute prediction of a model on a dataloader

    Args:
        model (nn.Module): Model to use for prediction
        dataloader (DataLoader): Dictionnary based dataloader
        mode (str) : Dataset mode

    Returns:
        pd.DataFrame: results dataframe containing "pred", "identifier" and "label
    """
    if cuda:
        model = model.cuda().eval()
    else:
        model = model.cpu().eval()

    preds = []
    labels = []
    ids = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm.tqdm(dataloader)):
            if cuda:
                batch["data"] = batch["data"].cuda()
            prediction = model.predict_step(batch, idx)
            prediction = prediction.cpu()

            preds += prediction.tolist()
            labels += batch[label].tolist()
            ids += batch["identifier"]
            torch.cuda.empty_cache()

    full = pd.DataFrame(columns=["pred"])
    full["pred"] = preds
    full["identifier"] = ids
    full["label"] = labels
    full["mode"] = mode
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
        logging.warning("Logging pretrain model")
        comet_logger = pl_module.logger
        comet_logger.experiment.log_model(
            name=pl_module.model.__class__.__name__, file_or_folder=self.best_model_path
        )
