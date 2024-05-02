import os

import torch 
from monai.data import CacheDataset
import pandas as pd

class TrainMrArt(CacheDataset):
    def __init__(self, transform, prefix:str):
        self.files = pd.read_csv("train.csv")
        self.files["data"] = prefix + self.files["data"]
        super().__init__(self.files.to_dict("records"), transform)
    
    @classmethod
    def lab(cls, transform):
        return cls(transform, "/home/at70870/narval/scratch/")
    
    @classmethod
    def narval(cls, transform):
        return cls(transform, "/home/cbricout/scratch/")