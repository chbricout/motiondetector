from monai.transforms import (Compose, LoadImaged,Orientationd, ScaleIntensityd, CropForegroundd, Resized,RandFlipd,RandRotated)


def threshold_one(x):
    return x >= 0.01

class Preprocess(Compose):
    def __init__(self, final_size=(160,192,160)):
        super().__init__(
   
        [
            LoadImaged(keys="data",ensure_channel_first=True, image_only=True),
            Orientationd(keys="data",axcodes="RAS"),
            ScaleIntensityd(keys="data", minv=0, maxv=1),
            CropForegroundd(source_key="data", keys="data",select_fn=threshold_one, allow_smaller=True),
            Resized(keys="data",spatial_size=final_size),
        ]
    )

class Augment(Compose):
  def __init__(self, spatial_axis=0, range_rotate=0.2):
        super().__init__(
        [
            RandFlipd(keys="data",prob=0.5, spatial_axis=spatial_axis),
            RandRotated(keys="data",prob=0.7, range_x=range_rotate, range_y=range_rotate, range_z=range_rotate),
        ])