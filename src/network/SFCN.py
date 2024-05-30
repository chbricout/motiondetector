import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution

from src.network.archi import ClassifierBase

class SFCNBlock(nn.Module): 
    def __init__(
        self,
        kernel_size,
        in_channel,
        out_channel,
        pool=True
    ):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm3d(out_channel),
        )
        if pool:
            self.block.append(nn.MaxPool3d(2,2))
        self.block.append(nn.ReLU())

    def forward(self, x):
        return self.block(x)
    
class SFCNHeadBlock(nn.Sequential): 
    def __init__(
        self,
        pool_size,
        in_channel,
        out_channel,
    ):
        super().__init__(
            nn.AvgPool3d(pool_size),
            nn.Dropout(p=0.5),
            nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, padding="same"),
            nn.Flatten()
        )




class SFCNModel(ClassifierBase):
    """
    Implementation of the model from Han Peng et al. in "Accurate brain age prediction with lightweight deep neural networks"
    https://doi.org/10.1016/j.media.2020.101871
    """
    def __init__(
        self,
        in_channel,
        im_shape,
        output_class=1,
        run_name="",
        lr=1e-5,
        mode="CLASS"
    ):
        super().__init__()

        self.im_shape = im_shape
        self.lr = lr
        self.run_name = run_name
        self.mode=mode
        self.encoder = nn.Sequential(
            SFCNBlock(3, in_channel, 32),
            SFCNBlock(3, 32, 64),
            SFCNBlock(3, 64, 128),
            SFCNBlock(3, 128, 256),
            SFCNBlock(3, 256, 256),
            SFCNBlock(1, 256, 64, pool=False),

        )

        self.im_shape = im_shape
        shape_like = (1, *im_shape)
        self.out_encoder = self.encoder(torch.empty(shape_like)).shape
        print(self.out_encoder)
       

        self.test_to_plot = None

        self.label = []
        self.classe = []
        self.save_hyperparameters()
        if self.mode=="CLASS":
            self.change_output_num(output_class)

        elif self.mode=="REGR":
            self.label_loss = nn.MSELoss()
            self.classifier = SFCNHeadBlock(self.out_encoder[2:] , 64,1)
            self.classifier.add_module("flatten_out", nn.Flatten(start_dim=0))


    def encode_forward(self, input):
        z = self.encoder(input)
        return z
    
    def change_output_num(self, num:int):
        self.classifier = SFCNHeadBlock(self.out_encoder[2:] , 64,num)
        if num == 1:
            self.classifier.add_module("flatten_out", nn.Flatten(start_dim=0))
            self.label_loss = nn.BCEWithLogitsLoss()
        elif num > 1:
            self.label_loss = nn.CrossEntropyLoss()


    def classify_emb(self, z):
        return self.classifier(z)

    def forward(self, x):
        z = self.encode_forward(x)
        classe = self.classify_emb(z)
        return [ z, classe]

   