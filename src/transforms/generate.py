"""Module to generate synthetic motion data"""

from collections import defaultdict
import json
from numbers import Number
import os
from typing import Callable, Dict
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    CenterSpatialCropd,
    RandomizableTransform,
    SaveImage,
    Transform,
)
from monai.losses.ssim_loss import SSIMLoss
from torchio.transforms import (
    RandomElasticDeformation,
    RandomGamma,
    RandomFlip,
    RandomBiasField,
)
from torchio import Motion, Subject
import torchio as tio
from scipy.spatial.transform import Rotation
import numpy as np
from src.motion.motion_magnitude import quantifyMotion


def get_affine(rot: np.ndarray, transl: np.ndarray) -> np.ndarray:
    """Create affine matrix from rotation and translation vector

    Args:
        rot (np.ndarray): rotation vector
        transl (np.ndarray): translation vector

    Returns:
        np.ndarray: affine matrix
    """
    rot_mat = Rotation.from_rotvec(rot, True).as_matrix()
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = rot_mat
    affine_matrix[:3, 3] = transl
    return affine_matrix


def get_matrices(transf_hist: Motion) -> np.ndarray:
    """Return affine matrices from an applied motion transform

    Args:
        transf_hist (Motion): Motion tranform used to modify a volume

    Returns:
        np.ndarray: affine matrices
    """
    rotations = transf_hist.degrees["data"]
    translations = transf_hist.translation["data"]
    affine_matrices = [np.eye(4)]
    for rot, transl in zip(rotations, translations):
        affine_matrices.append(get_affine(rot, transl))
    return affine_matrices


def get_motion_dist(affine_matrices: np.ndarray) -> float:
    """Compute average motion using the code from Pollak, C. et al.

    Args:
        affine_matrices (np.ndarray): list of successive affine matrices

    Returns:
        float: motion quantification
    """
    dist = quantifyMotion(affine_matrices)
    return np.array(dist).mean()


class CustomMotion(tio.transforms.RandomMotion, RandomizableTransform):
    """
    Adaptation of torchIO RandomMotion to generate volume with a goal quantified motion
    We use it to have a more uniform label distribution in the synthetic dataset
    """

    def __init__(
        self,
        goal_motion: float,
        tolerance: float = 0.02,
        num_transforms_range: tuple[int, int] = (4, 8),
    ):
        """Randomly generate a motion in the range [goal_motion-tolerance, goal_motion+tolerance]

        Args:
            goal_motion (float): quantify motion wanted
            tolerance (float, optional): acceptable tolerance. Defaults to 0.02.
        """
        self.transform_degrees = self.R.uniform(0, np.min((goal_motion / 3, 1)))
        self.goal_motion = goal_motion
        self.num_transforms = self.R.randint(*num_transforms_range)
        self.tolerance = tolerance

        super().__init__(
            self.transform_degrees,
            goal_motion,
            self.num_transforms,
            image_interpolation="bspline",
        )

    def apply_transform(self, subject: Subject) -> Subject:
        """Same transformation as torchIO RandomMotion, retry until goal motion is produced

        Args:
            subject (Subject): torchIO subject to use

        Returns:
            Subject: Subject with modified volumes
        """
        arguments: Dict[str, dict] = defaultdict(dict)
        for name, image in self.get_images_dict(subject).items():
            motion_mm = -1
            retry = 0
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
                retry += 1

            arguments["times"][name] = times_params
            arguments["degrees"][name] = degrees_params
            arguments["translation"][name] = translation_params
            arguments["image_interpolation"][name] = self.image_interpolation
        transform = tio.transforms.Motion(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        assert isinstance(transformed, Subject)
        return transformed


class Preprocess(Compose):
    """Transform to prepocess MRI volume before synthetic motion generation"""

    def __init__(self):
        self.tsf = [
            LoadImaged(keys="data", ensure_channel_first=True, image_only=True),
            Orientationd(keys="data", axcodes="RAS"),
            ScaleIntensityd(keys="data", minv=0, maxv=1),
        ]
        super().__init__(self.tsf)


class RandomScaleIntensityd(RandomizableTransform):
    """Transform to randomly scale intensity from min to
    max_range drawn with uniform distribution"""

    scaler: Callable

    def __init__(self, keys="data", minv=0, max_range=(0.9, 1.1)):
        """
        Args:
            keys (str, optional): keys to apply transform to. Defaults to "data".
            minv (int, optional): min value for rescale. Defaults to 0.
            max_range (tuple, optional): max value range for rescale. Defaults to (0.9, 1.1).
        """
        super().__init__(prob=1)
        self.max_range = max_range
        self.minv = minv
        self.keys = keys

    def randomize(self):
        """Create the scale transform with random max (uniform distribution)"""
        self.scaler = ScaleIntensityd(
            keys=self.keys, minv=self.minv, maxv=self.R.uniform(*self.max_range)
        )

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        self.randomize()
        return self.scaler(data)


class CreateSynthVolume(RandomizableTransform):
    """Transform to produce a synthetic volume using:
    - quantified random motion
    - random elastic deformation
    - random flip
    - random gamma
    - random bias field
    """

    apply_motion: bool
    apply_elastic: bool
    apply_flip: bool
    apply_corrupt: bool

    # Synthetic Parameters, not meant to be change often
    motion_prob: float = 0.9
    elastic_prob: float = 0.05
    flip_prob: float = 0.5
    corrupt_prob: float = 0.3
    goal_motion_range: tuple[float, float] = (0.05, 4.0)
    num_transforms_range: tuple[int, int] = (4, 8)
    tolerance: float = 0.01

    motion_tsf: CustomMotion
    goal_motion: float
    num_transforms: int

    def get_parameters(self) -> dict[str, Number | tuple[Number, Number]]:
        """Return a dictionnary summarizing all parameters for transformation

        Returns:
            dict[str, Number | tuple[Number, Number]]: Parameters for synthetic
            generation
        """
        return {
            "motion_prob": self.motion_prob,
            "elastic_prob": self.elastic_prob,
            "flip_prob": self.flip_prob,
            "corrupt_prob": self.corrupt_prob,
            "goal_motion_range": self.goal_motion_range,
            "num_transforms_range": self.num_transforms_range,
            "tolerance": self.tolerance,
        }

    def randomize(self):
        """Determine wich transform to apply"""
        super().randomize(None)
        self.apply_motion = self.R.rand() <= self.motion_prob
        self.apply_elastic = self.R.rand() <= self.elastic_prob
        self.apply_flip = self.R.rand() <= self.flip_prob
        self.apply_corrupt = self.R.rand() <= self.corrupt_prob
        if self.apply_motion:
            self.goal_motion = self.R.uniform(*self.goal_motion_range)

            self.motion_tsf = CustomMotion(
                self.goal_motion,
                tolerance=self.tolerance,
                num_transforms_range=self.num_transforms_range,
            )
        else:
            self.num_transforms = 0

    def __init__(
        self, prob: float = 1, do_transform: bool = True, elastic_activate: bool = True
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
        self.ssim = SSIMLoss(spatial_dims=3)

    def __call__(self, data):
        img = data["data"]
        self.randomize()
        if self.apply_flip:
            img = self.flip(img)
        if self.apply_elastic and self.elastic_activate:
            img = self.elastic_tsf(img)
        sub = Subject(data=tio.ScalarImage(tensor=img))

        before_motion = img
        if self.apply_motion:
            sub = self.motion_tsf(sub)
            affine_matrice = get_matrices(sub.get_composed_history()[0])
            motion_mm = get_motion_dist(affine_matrice)
        else:
            motion_mm = self.R.uniform(0, 0.05)

        img = sub["data"].data

        if self.apply_corrupt:
            img = self.corrupt(img)

        ssim_val = self.ssim(before_motion.unsqueeze(0), sub["data"].data.unsqueeze(0))
        return {
            "data": img,
            "motion_mm": motion_mm,
            "ssim_loss": ssim_val,
            "motion_binary": self.apply_motion,
            "identifier": data["identifier"],
        }


class FinalCrop(Compose):
    """Final transform before saving,
    apply crop and random intensity scale"""

    def __init__(self):
        tsfs = [
            CenterSpatialCropd(keys="data", roi_size=(160, 192, 160)),
            RandomScaleIntensityd(keys="data", minv=0, max_range=(0.9, 1.1)),
        ]
        super().__init__(tsfs)


class SyntheticPipeline(Transform):
    """Transform representing Synthetic generation process until volume storing"""

    def __init__(self, dataset_dir: str, mode="train", iteration=0):
        super().__init__()
        self.mode = mode
        self.iteration = iteration
        self.dataset_dir = dataset_dir
        self.base_path = f"{self.dataset_dir}/{self.mode}/"

        self.save = SaveImage(savepath_in_metadict=True, resample=False)
        self.synthetic_tsf = CreateSynthVolume(elastic_activate=True)
        self.process = Compose([self.synthetic_tsf, FinalCrop()])

    def __call__(self, element: dict[str, any]) -> dict[str, int | float | str | bool]:
        """Tranform and store the synthetic volume,
        returns all metadata to store as a dict

        Args:
            element (dict[str, any]): Element to process

        Returns:
            dict[str, int | float | str | bool]: dict containing
            the data to store as csv
        """
        synth = self.process(element)
        new_identifier = f"{element['identifier']}-{self.iteration}"
        path = os.path.join(self.base_path, new_identifier)
        self.save(synth["data"], filename=path)

        relative_path = os.path.relpath(path, os.path.dirname(self.dataset_dir))
        return {
            "data": relative_path+".nii.gz",
            "motion_mm": synth["motion_mm"],
            "ssim_loss": synth["ssim_loss"],
            "motion_binary": synth["motion_binary"],
            "identifier": new_identifier,
            "group": self.mode,
        }

    def save_parameters(self):
        """Save synthetic parameters for reproducibility purpose"""
        params = self.synthetic_tsf.get_parameters()
        with open(f"{self.dataset_dir}/parameters.json", "w") as file:
            json.dump(params, file)
