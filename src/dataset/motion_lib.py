from monai.data.dataset import CacheDataset
import pandas as pd

class MotionLib(CacheDataset):
    def __init__(self, transform=None, noise_level=None):
        self.files = pd.read_csv("motion_lib/scores.csv", index_col=0)
        if noise_level != None:
            self.files = self.files[self.files['label']==noise_level]
        super().__init__(self.files.to_dict("records"), transform)
