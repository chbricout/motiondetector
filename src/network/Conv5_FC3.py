import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution

from src.network.archi import ClassifierBase


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel):
        super().__init__(
            nn.Conv3d(in_channel, out_channel, 3, padding="same"),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
        )


class Conv5_FC3(ClassifierBase):
    """
    Implementation of the model from Simona Bottani et al. in "Automatic quality control of brain T1-weighted magnetic resonance images for a clinical data warehouse "
    https://doi.org/10.1016/j.media.2021.102219
    """

    def __init__(self, in_channel, im_shape, run_name="", lr=1e-5, mode="CLASS"):
        super().__init__()

        self.im_shape = im_shape
        self.lr = lr
        self.run_name = run_name
        self.mode = mode
        self.encoder = nn.Sequential(
            ConvBlock(in_channel, 8),
            ConvBlock(8, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )

        self.im_shape = im_shape
        shape_like = (1, *im_shape)
        self.out_encoder = self.encoder(torch.empty(shape_like))
        self.latent_size = self.out_encoder.numel()

        print(self.out_encoder)

        self.test_to_plot = None

        self.label = []
        self.classe = []
        self.save_hyperparameters()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(self.latent_size, 1300),
            nn.Linear(1300, 50),
        )

        if self.mode == "CLASS":
            self.label_loss = nn.BCEWithLogitsLoss()
        elif self.mode == "REGR":
            self.label_loss = nn.MSELoss()
        self.classifier.append(nn.Linear(50, 1))
        self.classifier.add_module("flatten_out", nn.Flatten(start_dim=0))

    def encode_forward(self, input):
        z = self.encoder(input)
        return z

    def classify_emb(self, z):
        return self.classifier(z)

    def forward(self, x):
        z = self.encode_forward(x)
        classe = self.classify_emb(z)
        return [ z, classe]
