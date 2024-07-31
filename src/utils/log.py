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
