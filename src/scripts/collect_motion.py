import pandas as pd
import sys
import os
import torch
import nibabel as nib
from itertools import product

sys.path.append(".")

from dataset.mrart.mrart_dataset import TrainMrArt, extract_sub
from monai.transforms import ScaleIntensity, CropForeground, Resize, Orientation


def get_noise(std, motion):
    mask = (motion <= 0) | (std <= 0)
    diff_map = motion - std
    diff_map[mask] = 0
    return diff_map


def load(path):
    img = nib.load(path)
    print(img)
    return torch.Tensor(img.get_fdata()), img


def threshold_one(x):
    return x >= 0.01


def process_noise(bids, score, noise_path, std_path, folder):
    name = f"noise-{score}-{bids}"
    if "standard" in std_path:
        name += "-standard"
    elif "headmotion1" in std_path:
        name += "-headmotion1"
    else:
        name += "-headmotion2"
    name += ".nii.gz"
    path_to_save = os.path.join(folder, name)
    print("Path to Save : ", path_to_save)

    scale = ScaleIntensity(0, 1)
    
    noisy, nif = load(noise_path)
    noisy =scale(noisy)
    std, _ = load(std_path)
    std = scale(std)
    noise = get_noise(std, noisy)

    nib.save(nib.Nifti1Image(noise, nif.affine, nif.header), path_to_save)
    return path_to_save, bids, score


if __name__ == "__main__":
    if os.path.exists("motion_lib"):
        os.removedirs("motion_lib")
    export = []
    qc_ds = TrainMrArt.lab()
    data = qc_ds.files
    data["bids"] = data["data"].apply(extract_sub)
    min_score = qc_ds.files.loc[qc_ds.files["label"] == 0]
    usable = data[data["bids"].isin(min_score["bids"])]
    usable = usable[usable["label"] > 0]

    os.makedirs("motion_lib")

    for index, sain in min_score.iterrows():
        print(f"using sub : {sain['bids']}")
        sub_files = data[data["bids"] == sain["bids"]]
        noise = sub_files[sub_files["label"] > 0]
        for i2, noisy in noise.iterrows():
            print(
                f"Getting noise out of file : {noisy['data']}, with ref {sain['data']}"
            )
            print(noisy)

            res = process_noise(
                noisy["bids"], noisy["label"], noisy["data"], sain["data"], "motion_lib"
            )
            export.append(res)
    pd.DataFrame(export, columns=["data", "bids_name", "label"]).to_csv(
        os.path.join("motion_lib", "scores.csv")
    )
