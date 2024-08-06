import sys

from src.config import IM_SHAPE

sys.path.append(".")
import torch
import pandas as pd
import numpy as np
import comet_ml
from monai.data.dataloader import DataLoader
from tqdm import tqdm
from src.dataset.mrart.mrart_dataset import ValMrArt
from src.network.archi import Model
from src.network.utils import parse_model
from src.transforms.load import FinetuneTransform


def bincount2d(arr, bins=None):
    if bins is None:
        bins = np.max(arr) + 1
    count = np.zeros(shape=[len(arr), bins], dtype=np.int64)
    indexing = (np.ones_like(arr).T * np.arange(len(arr))).T
    np.add.at(count, (indexing, arr), 1)

    return count


def evaluate_mcdropout(
    model: Model, dataloader, experiment, label="label", n_samples=100
):
    model.mc_dropout()
    res = []
    labels : list[int] = []
    with torch.no_grad():
        for _ in tqdm(range(n_samples)):
            sample_pred = []
            labels = []
            for idx, batch in enumerate(dataloader):
                batch["data"] = batch["data"].cuda()
                labels += batch[label].tolist()
                sample_pred.append(torch.as_tensor(model.predict_step(batch, idx)))
                torch.cuda.empty_cache()
            res.append(torch.concat(sample_pred).unsqueeze(1))
    preds = torch.concat(res, 1).float()
    print(preds, labels)
    mean = torch.mean(preds, dim=1)
    std = torch.std(preds, dim=1)
    count = bincount2d(preds.int())

    print(mean, std, count)
    full_np = np.array([mean.tolist(), std.tolist(), labels])
    full = pd.DataFrame(full_np.T, columns=["mean", "std", "labels"])
    full["count"] = count.tolist()

    if experiment != None:
        experiment.log_table("mcdropout-res.csv", full)

    return full

def pretrain_mcdropout(
    pl_module, dataloader, experiment, label="label", n_samples=100
):
    pl_module.model.mc_dropout()
    res: list[torch.Tensor] = []
    labels: list[float | int] = []
    with torch.no_grad():
        for _ in tqdm(range(n_samples)):
            sample_pred = []
            labels = []
            for idx, batch in enumerate(dataloader):
                batch["data"] = batch["data"].cuda()
                labels += batch[label].tolist()
                sample_pred.append(torch.as_tensor(pl_module.predict_step(batch, idx)))
                torch.cuda.empty_cache()
            sample_pred_tensor = torch.concat(sample_pred)
            res.append(sample_pred_tensor.unsqueeze(1))
    preds = torch.concat(res, 1).float()
    mean = torch.mean(preds, dim=1)
    std = torch.std(preds, dim=1)

    print(mean, std)
    full_np = np.array([mean.tolist(), std.tolist(), labels])
    full = pd.DataFrame(full_np.T, columns=["mean", "std", "labels"])

    if experiment != None:
        experiment.log_table("mcdropout-res.csv", full)

    return full

if __name__ == "__main__":
    PROJECT_NAME = "estimate-motion"
    DATASET = ValMrArt
    NUM_OUT = 3
    NUM_COUNT = 3
    EXP_TO_TRY = [
        "Conv5_FC3-4",
        "Conv5_FC3-2",
        "Conv5_FC3-1",
        "SFCN-1",
        "ViT-5",
        "ViT-4",
        "SFCN-2",
        "ViT-1",
        "Conv5_FC3-3",
        "Conv5_FC3-5",
        "SFCN-3",
        "ViT-3",
        "ViT-2",
        "SFCN-5",
        "SFCN-4",
        "SERes-3",
        "SERes-1",
        "SERes-5",
        "SERes-2",
        "SERes-4",
    ]
    api = comet_ml.api.API(
        api_key="WmA69YL7Rj2AfKqwILBjhJM3k",
    )
    load_tsf = FinetuneTransform()

    ds = DATASET.narval(load_tsf)
    val_loader = DataLoader(
        ds,
        batch_size=20,
        pin_memory=True,
    )

    for exp_name in tqdm(EXP_TO_TRY, desc="Processing experiments"):
        exp: comet_ml.APIExperiment = api.get("mrart", PROJECT_NAME, exp_name)

        model_name = exp.get_parameters_summary("model")["valueCurrent"]
        model_class = parse_model(model_name)
        net = model_class(
            IM_SHAPE,
            num_classes=NUM_OUT,
        )

        if len(exp.get_parameters_summary("run_num")) > 0:
            run_num = exp.get_parameters_summary("run_num")["valueCurrent"]
            exp.download_model(
                model_name,
                output_path=f"/home/cbricout/scratch/{PROJECT_NAME}-{run_num}/{model_name}",
            )
            net.load_state_dict(
                torch.load(
                    f"/home/cbricout/scratch/{PROJECT_NAME}-{run_num}/{model_name}/model-data/comet-torch-model.pth"
                )
            )
        else:
            exp.download_model(
                model_name,
                output_path=f"/home/cbricout/scratch/{PROJECT_NAME}/{model_name}",
            )
            net.load_state_dict(
                torch.load(
                    f"/home/cbricout/scratch/{PROJECT_NAME}/{model_name}/model-data/comet-torch-model.pth"
                )
            )
        net = net.cuda()

        print("processing : ", exp.get_name())
        evaluate_mcdropout(net, val_loader, exp)
