import torch.nn as nn
from monai.networks.blocks import Convolution, ResidualUnit


class ConvModule(nn.Module):
    def __init__(
        self,
        conv_kernel,
        in_channel,
        out_channel,
        stride,
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





class DeConvModule(nn.Module):
    def __init__(
        self,
        conv_kernel,
        in_channel,
        out_channel,
        stride=2,
        act="RELU",
    ):
        super().__init__()
        padding = conv_kernel // 2
        self.deconv_in = Convolution(
            spatial_dims=3,
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=conv_kernel,
            strides=stride,
            padding=padding,
            is_transposed=True,
            norm="BATCH",
            act=act,
        )
        self.deconv_mid = Convolution(
            spatial_dims=3,
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=conv_kernel,
            norm="BATCH",
            strides=1,
            padding=conv_kernel // 2,
            act=act,
        )

    def forward(self, x):
        y = self.deconv_in(x)
        y = self.deconv_mid(y)
        return y


class Classifier(nn.Sequential):
    def __init__(self, input_size, output_size=3, dropout_rate=0.5):
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        super().__init__(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.input_size, output_size),
        )
