import torch.nn as nn
from collections.abc import Sequence
from monai.networks.blocks import Convolution
from src.network.archi import Model, Encoder, Classifier


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        stride,
        conv_kernel=3,
        act="RELU",
    ):
        super().__init__()
        self.key = f"{in_channel}-{out_channel}"
        padding = conv_kernel // 2
        self.conv_in = Convolution(
            spatial_dims=3,
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=conv_kernel,
            padding=padding,
            strides=stride,
            norm="BATCH",
            act=act,
        )
        self.conv_mid = Convolution(
            spatial_dims=3,
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=conv_kernel,
            strides=1,
            padding=padding,
            norm="BATCH",
            act=act,
        )

    def forward(self, x):
        y = self.conv_in(x)
        y = self.conv_mid(y)

        return y


class CNNEncoder(Encoder):
    def __init__(self, im_shape: Sequence, dropout_rate: float):
        super().__init__(im_shape=im_shape, dropout_rate=dropout_rate)
        self.convs = nn.Sequential(
            ConvModule(1, 32, 2),
            ConvModule(32, 64, 2),
            ConvModule(64, 128, 2),
            ConvModule(128, 256, 2),
            ConvModule(256, 512, 2),
        )

    def forward(self, x):
        return self.convs(x)


class CNNClassifier(Classifier):
    input_size: int

    def __init__(self, input_size: int, num_classes: int, dropout_rate: float):
        super().__init__(input_size, num_classes, dropout_rate)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout_rate),
        )
        self.output_layer = nn.Linear(self.input_size, self.num_classes)

    def change_output_num(self, num_classes: int):
        self.num_classes = num_classes
        self.output_layer = nn.Linear(self.input_size, self.num_classes)


class CNNModel(Model):
    def __init__(self, im_shape: Sequence, num_classes: int, dropout_rate: float):
        super().__init__(
            im_shape=im_shape, num_classes=num_classes, dropout_rate=dropout_rate
        )
        self.encoder = CNNEncoder(self.im_shape, self.dropout_rate)
        self.classifier = CNNClassifier(
            self.encoder.latent_size, self.num_classes, self.dropout_rate
        )
