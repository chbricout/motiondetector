import re
from monai.data.dataset import Dataset, CacheDataset
import pandas as pd

def extract_sub(path: str):
    match_re = ".*(sub-[0-9A-Za-z]+).*"
    match_res = re.match(match_re, path)
    if match_res:
        return match_res.group(1)
    return ""

class TrainHCPEP(CacheDataset):
    def __init__(self, transform=None, prefix: str = ""):
        self.files = pd.read_csv("src/dataset/hcpep/pretrain.csv", index_col=0)
        self.files = self.files[self.files['group']=="train"]
        self.files["data"] = prefix + self.files["data"]
        super().__init__(self.files.to_dict("records"), transform, num_workers=10)

    @classmethod
    def lab(cls, transform=None):
        return cls(transform, "/home/at70870/narval/scratch/")

    @classmethod
    def narval(cls, transform=None):
        return cls(transform, "/home/cbricout/scratch/")


class ValHCPEP(CacheDataset):
    def __init__(self, transform=None, prefix: str = ""):
        self.files = pd.read_csv("src/dataset/hcpep/pretrain.csv", index_col=0)
        self.files = self.files[self.files['group']=="val"]
        self.files["data"] = prefix + self.files["data"]
        super().__init__(self.files.to_dict("records"), transform, num_workers=10)

    @classmethod
    def lab(cls, transform=None):
        return cls(transform, "/home/at70870/narval/scratch/")

    @classmethod
    def narval(cls, transform=None):
        return cls(transform, "/home/cbricout/scratch/")


class TestHCPEP(Dataset):
    def __init__(self, transform=None, prefix: str = ""):
        self.files = pd.read_csv("src/dataset/hcpep/pretrain.csv", index_col=0)
        self.files = self.files[self.files['group']=="test"]
        self.files["data"] = prefix + self.files["data"]
        super().__init__(self.files.to_dict("records"), transform)

    @classmethod
    def lab(cls, transform=None):
        return cls(transform, "/home/at70870/narval/scratch/")

    @classmethod
    def narval(cls, transform=None):
        return cls(transform, "/home/cbricout/scratch/")
