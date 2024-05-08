from monai.transforms import (
    RandomizableTransform,
    LoadImaged,
    Resize,
    Orientationd,
    Compose,
)
import torch
import numpy as np

from src.dataset.motion_lib import MotionLib


class MotionTsfd(RandomizableTransform):

    def randomize(self):
        super().randomize(None)
        self.motion_level = self.R.choice((0, 1, 2))

    def __init__(self, prob: float = 1, do_transform: bool = True, noisy=False):
        super().__init__(prob, do_transform)
        self.noisy = noisy
        self.pipeline = Compose(
            [
                LoadImaged(keys="data", ensure_channel_first=True, image_only=True),
                Orientationd(keys="data", axcodes="RAS"),
            ]
        )
        self.motion_lib = {
            1: MotionLib(self.pipeline, 1),
            2: MotionLib(self.pipeline, 2),
        }
        self.resize = Resize((160, 192, 160))

    def __call__(self, data):
        img = data["data"]
        bbx_start = data["foreground_start_coord"]
        bbx_end = data["foreground_end_coord"]
        self.randomize()
        if self.noisy:
            noise = torch.FloatTensor(
                self.R.randn(*img.shape) * np.sqrt(self.R.uniform(0, 0.001))
            )
        else:
            noise = torch.zeros_like(img)
        final = img
        if self.motion_level > 0:
            file_noise = self.R.choice(len(self.motion_lib[self.motion_level]))

            noise_motion = self.motion_lib[self.motion_level][file_noise]["data"]
            noise_motion = noise_motion[
                :,
                bbx_start[0] : bbx_end[0],
                bbx_start[1] : bbx_end[1],
                bbx_start[2] : bbx_end[2],
            ]
            noise_motion = self.resize(noise_motion)
            bin_vol = img <= 0
            noise_motion[bin_vol] = 0
            noise_motion = noise_motion.abs().clip(0,1)
            noise = noise + noise_motion

        final = img + noise

        return {"data": final.as_tensor(), "noise": noise, "label": self.motion_level}
