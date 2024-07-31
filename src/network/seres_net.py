import torch.nn as nn
from collections.abc import Sequence
from src.network.archi import Encoder, Classifier, Model


class SqueezeNExcite(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.module = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch, channel = x.shape[:2]
        y = self.pool(x).view(batch, channel)
        w = self.module(y).view(batch, channel, 1, 1, 1)
        return x * w


class SEResModule(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
    ):
        super().__init__()

        self.key = f"{in_channel}-{out_channel}"
        self.main_path = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, 3, padding="same"),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
            nn.Conv3d(out_channel, out_channel, 3, padding="same"),
            nn.BatchNorm3d(out_channel),
            SqueezeNExcite(out_channel),
            nn.ReLU(),
        )
        self.res_path = nn.Conv3d(in_channel, out_channel, 1, padding="same")
        self.out = nn.Sequential(nn.ReLU(), nn.MaxPool3d(2, 2))

    def forward(self, x):
        main = self.main_path(x)
        res = self.res_path(x)

        return self.out(main + res)


class SEResEncoder(Encoder):
    _latent_size: int

    def __init__(self, im_shape: Sequence, dropout_rate: float):
        super().__init__(im_shape=im_shape, dropout_rate=dropout_rate)
        self.convs = nn.Sequential(
            SEResModule(1, 8),
            SEResModule(8, 16),
            SEResModule(16, 32),
            SEResModule(32, 64),
            SEResModule(64, 128),
        )

    def forward(self, x):
        return self.convs(x)


class SEResClassifier(Classifier):
    input_size: int

    def __init__(self, input_size: int, num_classes: int, dropout_rate: float):
        super().__init__(input_size, num_classes, dropout_rate)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )
        self.output_layer = nn.Linear(256, self.num_classes)

    def change_output_num(self, num_classes: int):
        self.num_classes = num_classes
        self.output_layer = nn.Linear(256, self.num_classes)


class SEResModel(Model):
    def __init__(self, im_shape: Sequence, num_classes: int, dropout_rate: float):
        super().__init__(
            im_shape=im_shape, num_classes=num_classes, dropout_rate=dropout_rate
        )
        self.encoder = SEResEncoder(self.im_shape, self.dropout_rate)
        self.classifier = SEResClassifier(
            self.encoder.latent_size, self.num_classes, self.dropout_rate
        )
