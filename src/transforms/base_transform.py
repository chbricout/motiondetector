from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    CropForegroundd,
    Resized,
    RandFlipd,
    RandRotated,
)
from src.transforms.soft_label import ProgressiveSoftEncode, FloatLabel



def threshold_one(x):
    return x >= 0.01


class Preprocess(Compose):
    def __init__(self, final_size=(160, 192, 160), soft_labeling=False, mode:str="CLASS"):
        self.tsf =  [
                LoadImaged(keys="data", ensure_channel_first=True, image_only=True),
                Orientationd(keys="data", axcodes="RAS"),
                ScaleIntensityd(keys="data", minv=0, maxv=1),
                CropForegroundd(
                    source_key="data",
                    keys="data",
                    select_fn=threshold_one,
                    allow_smaller=True,
                ),
                Resized(keys="data", spatial_size=final_size),
            ]
        if soft_labeling:
            self.tsf.append(ProgressiveSoftEncode(keys="label"))
        elif mode == "REGR":
            self.tsf.append(FloatLabel(keys="label"))
        super().__init__(
           self.tsf
        )


class Augment(Compose):
    def __init__(self, keys :str | list[str]= "data", spatial_axis=0, range_rotate=0.2):
        super().__init__(
            [
                RandFlipd(keys=keys, prob=0.5, spatial_axis=spatial_axis),
                RandRotated(
                    keys=keys,
                    prob=0.7,
                    range_x=range_rotate,
                    range_y=range_rotate,
                    range_z=range_rotate,
                ),
            ]
        )
