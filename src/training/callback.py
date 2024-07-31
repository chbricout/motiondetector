import logging
import shutil
import tempfile
import comet_ml
from matplotlib import pyplot as plt
from monai.data.dataloader import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import pandas as pd
import seaborn as sb
import torch
from sklearn.metrics import r2_score
from src.dataset.ampscz.ampscz_dataset import FinetuneValAMPSCZ
from src.dataset.mrart.mrart_dataset import ValMrArt
from src.utils.mcdropout import evaluate_mcdropout, pretrain_mcdropout
from src.transforms.load import FinetuneTransform, ToSoftLabel


def get_correlations(model, exp: comet_ml.BaseExperiment):
    load_tsf = FinetuneTransform()
    for dataset in (ValMrArt, FinetuneValAMPSCZ):
        dl = DataLoader(dataset.narval(load_tsf))
        res = get_pred_from_pretrain(model, dl)
        exp.log_metric(f"{dataset.__name__}-r2", r2_score(res["label"], res["mean"]))
        exp.log_table(f"{dataset.__name__}-pred.csv", res)

        with tempfile.NamedTemporaryFile() as img_file:
            fig = get_calibration_curve(res["mean"], res["label"], hue=res["std"])
            fig.savefig(img_file)
            exp.log_image(
                img_file.name,
                f"{dataset.__name__}-calibration",
            )


def get_pred_from_pretrain(model, dataloader: DataLoader):
    model = model.cuda().eval()
    softLabel = ToSoftLabel.baseConfig()
    means = []
    stds = []
    labels = []
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            volume = batch["data"].cuda()
            prediction = model(volume)
            prediction = prediction.cpu()
            mean, std = softLabel.softLabelToMeanStd(prediction)
            means += mean.tolist()
            stds += std.tolist()
            labels += batch["label"].tolist()
            torch.cuda.empty_cache()

    full = pd.DataFrame(columns=["mean"])
    full["mean"] = means
    full["std"] = stds
    full["file"] = dataloader.dataset.files["data"].apply(lambda x: x.split("/")[-1])
    full["label"] = labels
    return full


def get_calibration_curve(prediction, label, hue=None):
    fig = plt.figure(figsize=(6, 5))
    sb.scatterplot(x=label, y=prediction, hue=hue)
    plt.plot([0, 3], [0, 3], "r")
    plt.xlabel("Correct Label")
    plt.ylabel("Estimated Label")
    return fig


class FinetuneCallback(ModelCheckpoint):
    def on_fit_end(self, trainer: Trainer, pl_module):
        comet_logger = pl_module.logger
        comet_logger.experiment.log_model(
            name=pl_module.model_class.__name__, file_or_folder=self.best_model_path
        )
        best_net = pl_module.__class__.load_from_checkpoint(self.best_model_path)
        

        evaluate_mcdropout(best_net, trainer.val_dataloaders, comet_logger.experiment)


class PretrainCallback(ModelCheckpoint):
    def on_fit_end(self, trainer: Trainer, pl_module):
        logging.info("Logging pretrain model")
        comet_logger = pl_module.logger
        comet_logger.experiment.log_model(
            name=pl_module.model_class.__name__, file_or_folder=self.best_model_path
        )

        best_net = pl_module.__class__.load_from_checkpoint(self.best_model_path)

        logging.info("Running correlation on pretrain")
        get_correlations(best_net, comet_logger.experiment)

        logging.info("Running dropout on pretrain")
        pretrain_mcdropout(
            best_net,trainer.val_dataloaders, comet_logger.experiment, "motion_mm"
        )

        logging.info("Removing Checkpoints")
        shutil.rmtree(trainer.default_root_dir)
