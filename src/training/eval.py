"""Module defining the callback used at on the Pretraining and Finetuning task
 and any needed function"""

import comet_ml
import pandas as pd
import torch
import tqdm
from monai.data.dataloader import DataLoader

from src.training.pretrain_logic import PretrainingTask


def get_pred_from_pretrain(
    model: PretrainingTask,
    dataloader: DataLoader,
    mode: str = "test",
    label: str = "label",
    cuda=True,
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
    print(len(preds), len(ids), len(labels))
    full["pred"] = preds
    full["identifier"] = ids
    full["label"] = labels
    full["mode"] = mode
    return full
