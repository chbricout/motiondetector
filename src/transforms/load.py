from typing import Dict, Hashable, Mapping, Union
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    CenterSpatialCropd,
    Transform,
    MapTransform,
)
from scipy.stats import norm
import numpy as np
import torch

from src.config import BIN_RANGE, BIN_STEP, N_BINS


class ToSoftLabel(MapTransform):
    def __init__(
        self,
        keys,
        backup_keys,
        bin_range: tuple,
        bin_step: float,
        soft_label: bool = True,
        require_grad=False,
    ):
        """
        adapted from https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain/blob/master/dp_model/dp_utils.py

        v,bin_centers = number2vector(x,bin_range,bin_step,sigma)
        bin_range: (start, end), size-2 tuple
        bin_step: should be a divisor of |end-start|
        soft_label:True 'soft label', v is vector else 'hard label', v is index
        debug: True for error messages.
        """
        super().__init__(keys)
        if isinstance(backup_keys, tuple):
            self.backup_keys = backup_keys
        else:
            self.backup_keys = (backup_keys,)

        self.bin_start = bin_range[0]
        self.bin_end = bin_range[1]
        self.bin_length = self.bin_end - self.bin_start
        if not round(self.bin_length / bin_step, 5) % 1 == 0:
            raise ValueError("bin's range should be divisible by bin_step!")

        # self.bin_range = bin_range
        self.bin_step = bin_step
        self.soft_label = soft_label
        self.bin_number = int(round(self.bin_length / bin_step))
        self.bin_centers = (
            self.bin_start + float(bin_step) / 2 + bin_step * np.arange(self.bin_number)
        )

        if require_grad:
            self.bin_centers = torch.tensor(self.bin_centers, dtype=torch.float32)

    def __call__(
        self, data: Mapping[Hashable, Union[torch.Tensor]]
    ) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key, backup in zip(self.keys, self.backup_keys):
            if torch.is_tensor(d[key]):
                d[backup] = d[key].clone()
            else:
                d[backup] = d[key]

            d[key] = self.valueToSoftlabel(d[key])

        return d

    def valueToSoftlabel(self, x):
        if torch.is_tensor(x):
            was_tensor = True
            x = x.squeeze().numpy()
            assert len(x.shape) == 1 or len(x.shape) == 0
            x = x.tolist()
        else:
            was_tensor = False

        if not self.soft_label:
            x = np.array(x)
            i = np.floor((x - self.bin_start) / self.bin_step)
            i = i.astype(int)
            return i if not was_tensor else torch.tensor(i)
        else:
            if np.isscalar(x):
                v = np.zeros((self.bin_number,))
                for i in range(self.bin_number):
                    x1 = self.bin_centers[i] - float(self.bin_step) / 2
                    x2 = self.bin_centers[i] + float(self.bin_step) / 2
                    cdfs = norm.cdf(
                        [x1, x2], loc=x, scale=self.bin_length * 0.03
                    )  # TODO: test effects of sigma
                    v[i] = cdfs[1] - cdfs[0]
            else:
                v = np.zeros((len(x), self.bin_number))
                for j in range(len(x)):
                    for i in range(self.bin_number):
                        x1 = self.bin_centers[i] - float(self.bin_step) / 2
                        x2 = self.bin_centers[i] + float(self.bin_step) / 2
                        cdfs = norm.cdf(
                            [x1, x2], loc=x[j], scale=self.bin_length * 0.03
                        )
                        v[j, i] = cdfs[1] - cdfs[0]

            return v if not was_tensor else torch.tensor(v)

    def get_probs(self, x):
        if torch.sum(x) > 1.0:
            x = torch.nn.functional.log_softmax(x.squeeze())

        return torch.exp(x)

    def softLabelToHardLabel(self, x):
        pred = self.get_probs(x) @ self.bin_centers

        return pred

    def softLabelToMeanStd(self, x):
        prob = self.get_probs(x)
        if torch.is_tensor(prob):
            prob = prob.numpy()

        mean = prob @ self.bin_centers

        if torch.is_tensor(mean):
            mean = mean.numpy()

        if len(x.shape) == 1:
            var = np.average((self.bin_centers - mean) ** 2, weights=prob)
        else:
            var = np.average(
                (self.bin_centers - mean[np.newaxis, :].T) ** 2, weights=prob, axis=1
            )
        return mean, np.sqrt(var)

    @staticmethod
    def baseConfig():
        return ToSoftLabel(
            keys="label",
            backup_keys="motion_mm",
            bin_range=BIN_RANGE,
            bin_step=BIN_STEP,
        )


class LoadSynth(Compose):
    def __init__(self, num_bins=N_BINS, bin_range=BIN_RANGE):
        bin_step = (bin_range[1] - bin_range[0]) / num_bins
        self.soft_label = ToSoftLabel(
            keys="label",
            backup_keys="motion_mm",
            bin_range=bin_range,
            bin_step=bin_step,
        )
        self.tsf = [
            LoadImaged(keys="data", ensure_channel_first=True, image_only=True),
            Orientationd(keys="data", axcodes="RAS"),
            self.soft_label,
        ]
        super().__init__(self.tsf)


class FinetuneTransform(Compose):
    def __init__(self):
        super().__init__(
            [
                LoadImaged(keys="data", ensure_channel_first=True, image_only=True),
                Orientationd(keys="data", axcodes="RAS"),
                CenterSpatialCropd(keys="data", roi_size=(160, 192, 160)),
                ScaleIntensityd(keys="data", minv=0, maxv=1),
            ]
        )
