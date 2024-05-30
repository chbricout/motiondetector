import torch
import torch.nn as nn

from src.network.archi import ClassifierBase


class ResConvModule(nn.Module):
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
        )
        self.res_path = nn.Conv3d(in_channel, out_channel, 1, padding="same")

        self.out = nn.Sequential(nn.ReLU(), nn.MaxPool3d(2, 2))

    def forward(self, x):
        main = self.main_path(x)
        res = self.res_path(x)

        return self.out(main + res)


class ResNetModel(ClassifierBase):
    def __init__(self, in_channel, im_shape,output_class=1, run_name="", lr=1e-5, mode="CLASS"):
        super().__init__()
        self.mode = mode
        self.im_shape = im_shape
        self.lr = lr
        self.run_name = run_name

        self.encoder = nn.Sequential(
            ResConvModule(in_channel, 8),
            ResConvModule(8, 16),
            ResConvModule(16, 32),
            ResConvModule(32, 64),
            ResConvModule(64, 128),
        )

        self.im_shape = im_shape
        shape_like = (1, *im_shape)
        self.out_encoder = self.encoder(torch.empty(shape_like))
        self.latent_size = self.out_encoder.numel()
        print(self.latent_size)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.latent_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
     

        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Dropout(0.5),
        #     nn.Linear(self.latent_size, 1300),
        #     nn.Linear(1300, 50),
        #        nn.Linear(50, 1)
        # )

        if self.mode == "CLASS":
            self.change_output_num(output_class)
        elif self.mode == "REGR":
            self.label_loss = nn.MSELoss()
        self.test_to_plot = None

        self.label = []
        self.classe = []
        self.save_hyperparameters()

    def change_output_num(self, num: int):
        if num == 1:
            self.classifier_output = nn.Sequential(
                nn.Linear(256, num), nn.Flatten(start_dim=0)
            )
            self.label_loss = nn.BCEWithLogitsLoss()
        elif num > 1:
            self.classifier_output = nn.Linear(256, num)
            self.label_loss = nn.CrossEntropyLoss()

    def encode_forward(self, input):
        z = self.encoder(input)
        return z

    def classify_emb(self, z):
        return self.classifier_output(self.classifier(torch.flatten(z, start_dim=1)))

    def forward(self, x):
        z = self.encode_forward(x)

        classe = self.classify_emb(z)
        return [z, classe]
