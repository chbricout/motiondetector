import logging
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage


def save_volume_as_gif(volume: torch.Tensor, file_path: str):
    convert = ToPILImage()
    imgs = [convert(img) for img in volume]

    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save(file_path, save_all=True, append_images=imgs[1:], duration=50, loop=0)
