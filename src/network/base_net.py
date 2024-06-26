import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution

from src.network.archi import ReconstructBase
from src.network.blocks import ConvModule, DeConvModule, Classifier


class BaselineModel(ReconstructBase):
    def __init__(
        self,
        in_channel,
        im_shape,
        output_class=1,
        act="RELU",
        kernel_size=3,
        run_name="",
        lr=1e-5,
        beta=1,
        use_decoder=True,
        dropout_rate=0.5,
        mode="CLASS"
    ):
        super().__init__()

        self.im_shape = im_shape
        self.lr = lr
        self.beta = beta
        self.use_decoder = use_decoder
        self.run_name = run_name
        self.dropout_rate=dropout_rate
        self.mode=mode
        self.encoder = nn.Sequential(
            ConvModule(kernel_size, in_channel, 32, 2, act=act),
            ConvModule(kernel_size, 32, 64, 2, act=act),
            ConvModule(kernel_size, 64, 128, 2, act=act),
            ConvModule(kernel_size, 128, 256, 2, act=act),
            ConvModule(kernel_size, 256, 512, 2, act=act),
         
        )

        self.im_shape = im_shape
        shape_like = (1, *im_shape)
        self.out_encoder = self.encoder(torch.empty(shape_like))
        self.latent_size = self.out_encoder.numel()
        print(self.latent_size)
        if self.use_decoder:
            self.decoder = nn.Sequential(
                DeConvModule(kernel_size, 512, 256, act=act),
                DeConvModule(kernel_size, 256, 128, act=act),
                DeConvModule(kernel_size, 128, 64, act=act),
                DeConvModule(kernel_size, 64, 32, act=act),
                DeConvModule(kernel_size, 32, in_channel, act=act),
            )


        self.recon_to_plot = None
        self.test_to_plot = None

        self.label = []
        self.classe = []
        self.save_hyperparameters()
        if self.mode=="CLASS":
            self.recon_loss = nn.MSELoss()
            self.change_output_num(output_class)

        elif self.mode=="REGR":
            self.label_loss = nn.MSELoss()
            self.recon_loss = nn.MSELoss()
            self.classifier = Classifier(self.latent_size, 1, self.dropout_rate)

    def change_output_num(self, num:int):
        self.classifier = Classifier(self.latent_size, num, self.dropout_rate)
        if num == 1:
            self.classifier.add_module("flatten_out", nn.Flatten(start_dim=0))       
            self.label_loss = nn.BCEWithLogitsLoss()
        elif num > 1:
            self.label_loss = nn.CrossEntropyLoss()

    def encode_forward(self, input):
        z = self.encoder(input)
        return z

    def decode_forward(self, z):
        res = self.decoder(z)
        return res

    def classify_emb(self, z):
        return self.classifier(torch.flatten(z, start_dim=1))

    def forward(self, x):
        z = self.encode_forward(x)
        if self.use_decoder:
            recon = self.decode_forward(z)
        else:
            recon = x
        classe = self.classify_emb(z)
        return [recon, z, classe]

   