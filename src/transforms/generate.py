from collections import defaultdict
from typing import Dict
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    CenterSpatialCropd,
    RandomizableTransform,
)
from torchio.transforms import (
    RandomElasticDeformation,
    RandomGamma,
    RandomFlip,
    RandomBiasField,
)
from torchio import Subject
import torchio as tio
from scipy.spatial.transform import Rotation
import numpy as np
from src.motion.motion_magnitude import quantifyMotion


def threshold_one(x):
    return x >= 0.01


def get_affine(rot, transl):
    rot_mat = Rotation.from_rotvec(rot, True).as_matrix()
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = rot_mat
    affine_matrix[:3, 3] = transl
    return affine_matrix


def get_matrices(transf_hist):
    rotations = transf_hist.degrees["data"]
    translations = transf_hist.translation["data"]
    affine_matrices = [np.eye(4)]
    for rot, transl in zip(rotations, translations):
        affine_matrices.append(get_affine(rot, transl))
    return affine_matrices


def get_motion_dist(affine_matrice):
    dist = quantifyMotion(affine_matrice)
    return np.array(dist).mean()

class CustomMotion(tio.transforms.RandomMotion, RandomizableTransform):
    def __init__(self, goal_motion, tolerance=0.02):
        self.transform_degrees = self.R.uniform(0, np.min((goal_motion / 3, 1)))
        self.goal_motion = goal_motion
        self.num_transforms = self.R.randint(1, 8)
        self.tolerance = tolerance

        super().__init__(
            self.transform_degrees,
            goal_motion,
            self.num_transforms,
            image_interpolation="bspline",
        )

    def apply_transform(self, subject: Subject) -> Subject:
        arguments: Dict[str, dict] = defaultdict(dict)
        for name, image in self.get_images_dict(subject).items():
            motion_mm = -1
            while (
                motion_mm > self.goal_motion + self.tolerance
                or motion_mm < self.goal_motion - self.tolerance
            ):
                params = self.get_params(
                    self.degrees_range,
                    self.translation_range,
                    self.num_transforms,
                    is_2d=image.is_2d(),
                )
                times_params, degrees_params, translation_params = params
                rotations = degrees_params
                translations = translation_params
                affine_matrices = [np.eye(4)]
                for rot, transl in zip(rotations, translations):
                    affine_matrices.append(get_affine(rot, transl))
                motion_mm = get_motion_dist(affine_matrices)

            arguments["times"][name] = times_params
            arguments["degrees"][name] = degrees_params
            arguments["translation"][name] = translation_params
            arguments["image_interpolation"][name] = self.image_interpolation
        transform = tio.transforms.Motion(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        assert isinstance(transformed, Subject)
        return transformed


class Preprocess(Compose):
    def __init__(self):
        self.tsf = [
            LoadImaged(keys="data", ensure_channel_first=True, image_only=True),
            Orientationd(keys="data", axcodes="RAS"),
            ScaleIntensityd(keys="data", minv=0, maxv=1),
        ]
        super().__init__(self.tsf)


class RandomScaleIntensityd(RandomizableTransform):
    def __init__(self, keys="data", minv=0, max_range=(0.9, 1.1)):
        super().__init__(prob=1)
        self.max_range = max_range
        self.minv = minv
        self.keys = keys

    def randomize(self):
        self.scaler = ScaleIntensityd(
            keys=self.keys, minv=self.minv, maxv=self.R.uniform(*self.max_range)
        )

    def __call__(self, data):
        self.randomize()
        return self.scaler(data)


class CreateSynthVolume(RandomizableTransform):
    def randomize(self):
        super().randomize(None)
        self.motion = self.R.rand() > 0.1
        self.elastic = self.R.rand() > 0.05
        self.can_flip = self.R.rand() > 0.5
        self.can_corrupt = self.R.rand() > 0.3
        if self.motion:
            self.goal_motion = self.R.uniform(0.1, 2)

            self.motion_tsf = CustomMotion(self.goal_motion, tolerance=0.01)
        else:
            self.num_transforms = 0

    def __init__(
        self, prob: float = 1, do_transform: bool = True, elastic_activate=True
    ):
        super().__init__(prob, do_transform)
        self.elastic_tsf = RandomElasticDeformation(
            num_control_points=7, max_displacement=8, image_interpolation="bspline"
        )

        self.elastic_activate = elastic_activate
        self.corrupt = Compose(
            [RandomGamma((-0.1, 0.1)), RandomBiasField(coefficients=(0.0, 0.35))]
        )
        self.flip = RandomFlip(0, flip_probability=1)
        self.goal_motion = 0

    def __call__(self, data):
        img = data["data"]
        self.randomize()
        if self.can_flip:
            img = self.flip(img)
        if self.elastic and self.elastic_activate:
            img = self.elastic_tsf(img)
        sub = Subject(data=tio.ScalarImage(tensor=img))

        if self.motion:
            sub = self.motion_tsf(sub)
            affine_matrice = get_matrices(sub.get_composed_history()[0])
            motion_mm = get_motion_dist(affine_matrice)
        else:
            motion_mm = self.R.uniform(0, 0.15)

        img = sub["data"].data

        if self.can_corrupt:
            img = self.corrupt(img)

        return {
            "data": img,
            "motion_mm": motion_mm,
            "sub_id": data["sub_id"],
            "ses_id": data["ses_id"],
        }


class FinalCrop(Compose):
    def __init__(self):
        tsfs = [
            CenterSpatialCropd(keys="data", roi_size=(160, 192, 160)),
            RandomScaleIntensityd(keys="data", minv=0, max_range=(0.9, 1.1)),
        ]
        super().__init__(tsfs)
