import os

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage
from monai.transforms.intensity.array import ScaleIntensity


def save_volume_as_gif(volume: torch.Tensor, file_path: str):
    convert = ToPILImage()
    scale_01 = ScaleIntensity(0, 1)
    scaled = scale_01(volume)
    imgs = [convert(img) for img in scaled]

    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save(file_path, save_all=True, append_images=imgs[1:], duration=50, loop=0)

def get_run_dir(project_name:str, run_name:str, narval:bool):
    if narval:
        root_dir = f"/home/cbricout/scratch/{project_name}"
    else :
        root_dir = f"/home/at70870/local_scratch/{project_name}"

    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    run_dir = f"{root_dir}/{run_name}"
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    return run_dir
