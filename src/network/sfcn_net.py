import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution
from collections.abc import Sequence
from src.network.archi import Encoder, Classifier, Model


class SFCNBlock(nn.Module):
    def __init__(self, kernel_size, in_channel, out_channel, pool=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.BatchNorm3d(out_channel),
        )
        if pool:
            self.block.append(nn.MaxPool3d(2, 2))
        self.block.append(nn.ReLU())

    def forward(self, x):
        return self.block(x)


class SFCNHeadBlock(nn.Sequential):
    def __init__(self, pool_size, in_channel, out_channel, dropout_rate):
        super().__init__(
            nn.AvgPool3d(pool_size),
            nn.Dropout(p=dropout_rate),
            nn.Conv3d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=1,
                padding="same",
            ),
            nn.Flatten(),
        )


class SFCNEncoder(Encoder):

    def __init__(self, im_shape: Sequence, dropout_rate: float):
        super().__init__(im_shape=im_shape, dropout_rate=dropout_rate)
        self.convs = nn.Sequential(
            SFCNBlock(3, 1, 32),
            SFCNBlock(3, 32, 64),
            SFCNBlock(3, 64, 128),
            SFCNBlock(3, 128, 256),
            SFCNBlock(3, 256, 256),
            SFCNBlock(1, 256, 64, pool=False),
        )

    def forward(self, x):
        return self.convs(x)




class SFCNClassifier(Classifier):
    def __init__(self, input_size: Sequence, num_classes: int, dropout_rate: float):
        super().__init__(input_size, num_classes, dropout_rate)

        self.classifier = nn.Identity()
        self.output_layer = SFCNHeadBlock(
            self.input_size, 64, self.num_classes, self.dropout_rate
        )

    def change_output_num(self, num_classes: int):
        self.num_classes = num_classes
        self.output_layer = SFCNHeadBlock(
            self.input_size, 64, self.num_classes, self.dropout_rate
        )


class SFCNModel(Model):
    """
    Implementation of the model from Han Peng et al. in "Accurate brain age prediction with lightweight deep neural networks"
    https://doi.org/10.1016/j.media.2020.101871
    """

    def __init__(self, im_shape: Sequence, num_classes: int, dropout_rate: float):
        super().__init__(
            im_shape=im_shape, num_classes=num_classes, dropout_rate=dropout_rate
        )
        self.encoder = SFCNEncoder(self.im_shape, self.dropout_rate)
        self.classifier = SFCNClassifier(
            self.encoder.latent_shape, self.num_classes, self.dropout_rate
        )
