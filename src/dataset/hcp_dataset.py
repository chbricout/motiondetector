import re
from monai.data.dataset import CacheDataset
import pandas as pd


class HCP(CacheDataset):
    def __init__(self, transform=None, prefix: str = ""):
        self.files = pd.read_csv("src/dataset/hcp.csv", index_col=0)
        self.files["data"] = prefix + self.files["data"]
        super().__init__(self.files.to_dict("records"), transform)

    @classmethod
    def lab(cls, transform=None):
        return cls(transform, "/home/at70870/narval/scratch/")

    @classmethod
    def narval(cls, transform=None):
        return cls(transform, "/home/cbricout/scratch/")
