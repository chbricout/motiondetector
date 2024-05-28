from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    CenterSpatialCropd,
    RandomizableTransform
)
from torchio.transforms import RandomMotion,RandomElasticDeformation
from src.transforms.soft_label import ProgressiveSoftEncode, FloatLabel



def threshold_one(x):
    return x >= 0.01


class Preprocess(Compose):
    def __init__(self, soft_labeling=False, mode:str="CLASS"):
        self.tsf =  [
                LoadImaged(keys="data", ensure_channel_first=True, image_only=True),
                Orientationd(keys="data", axcodes="RAS"),
                ScaleIntensityd(keys="data", minv=0, maxv=1),
            ]
        if soft_labeling:
            self.tsf.append(ProgressiveSoftEncode(keys="label"))
        super().__init__(
           self.tsf
        )



class CreateSynthVolume(RandomizableTransform):

    def randomize(self):
        super().randomize(None)
        self.motion = self.R.rand()>0.5
        self.elastic = self.R.rand()>0.2
        

    def __init__(self, prob: float = 1, do_transform: bool = True, elastic_activate=True):
        super().__init__(prob, do_transform)
        self.elastic_tsf = RandomElasticDeformation(num_control_points=7, max_displacement=7)
        self.motion_tsf = RandomMotion(degrees=[6,8],translation=[4,6], num_transforms=3)
        self.elastic_activate = elastic_activate

    def __call__(self, data):
        img = data["data"]
        self.randomize()
     
        if self.elastic and self.elastic_activate:
            img = self.elastic_tsf(img)
        if self.motion:
            img = self.motion_tsf(img)

        return {"data": img, "label": float(self.motion)}


class FinalCrop(CenterSpatialCropd):
    def __init__(self):
        super().__init__(keys="data",roi_size=(160, 192, 160)        )
