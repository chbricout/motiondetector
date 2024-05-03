import os
import torch
import torch.nn as nn
from monai.networks.blocks import Convolution
import lightning

from src.network.blocks import ConvModule, DeConvModule, Classifier
from src.training.log import save_volume_as_gif

class BaselineModel(lightning.LightningModule):
   

    def __init__(
        self,
        in_channel,
        im_shape,
        act="PRELU",
        kernel_size=5,
        run_name="",
        lr=1e-4,
        beta=0.1,
        use_decoder=True

    ):
        super().__init__()

        self.im_shape = im_shape
        self.lr=lr
        self.beta=beta
        self.use_decoder = use_decoder
        self.run_name = run_name

        self.encoder = nn.Sequential(
            ConvModule(kernel_size, in_channel, 32,  2, act=act),
            ConvModule(kernel_size, 32, 64,  2, act=act),
            ConvModule(kernel_size, 64, 128, 2, act=act),
            ConvModule(kernel_size, 128, 256,  2, act=act),
            ConvModule(kernel_size, 256, 512,  2, act=act),
            Convolution(3,512,3,kernel_size=kernel_size, padding=kernel_size//2, norm="BATCH"),
        )

        self.im_shape = im_shape
        shape_like = (1, *im_shape)
        self.out_encoder = self.encoder(torch.empty(shape_like))
        self.latent_size = self.out_encoder.numel()
        print(self.latent_size)
        if self.use_decoder :
            self.decoder = nn.Sequential(
                Convolution(3,3,512,kernel_size=kernel_size, padding=kernel_size//2, norm="BATCH"),
                DeConvModule(kernel_size, 512, 256,  act=act),
                DeConvModule(kernel_size, 256, 128,  act=act),
                DeConvModule(kernel_size, 128, 64,  act=act),
                DeConvModule(kernel_size, 64, 32,  act=act),
                DeConvModule(kernel_size, 32, in_channel,act=act),
            )

        self.classifier = Classifier(self.latent_size, 3)


        self.recon_to_plot = None
        self.test_to_plot = None

        self.label=[]
        self.classe=[]
        self.save_hyperparameters()

    def encode_forward(self, input):
        z =self.encoder(input)
        return z

    def decode_forward(self, z):
        res = self.decoder(z)
        return res

    def classify_emb(self, z):
        return self.classifier(torch.flatten(z, start_dim=1))

    def forward(self, x):
        z = self.encode_forward(x)
        if self.use_decoder :
            recon = self.decode_forward(z)
        else :
            recon = x
        classe = self.classify_emb(z)
        return [recon, z, classe]

    def training_step(self, batch, batch_idx):
        volume = batch['data']
        label = batch['label']
        recon_batch, emb, classe = self.forward(volume)
        # LOSS COMPUTE
        if self.use_decoder:
            recon_loss = torch.nn.functional.mse_loss(recon_batch, volume)
            label_loss = torch.nn.functional.cross_entropy(classe, label)
            model_loss_tot = recon_loss  + self.beta * label_loss
            self.log("train_recon_loss", recon_loss)
        else: 
            label_loss = torch.nn.functional.cross_entropy(classe, label)
            model_loss_tot =  label_loss
        self.log("train_loss", model_loss_tot)
        self.log("train_label_loss", label_loss)

        return model_loss_tot

    def validation_step(self, batch, batch_idx):
        ## VAE TESTING PHASE##
        # INFERENCE
        volume = batch['data']
        label = batch['label']
        recon_batch, emb, classe = self.forward(volume)
        # LOSS COMPUTE
        if self.use_decoder:
            recon_loss = torch.nn.functional.mse_loss(recon_batch, volume)
            label_loss = torch.nn.functional.cross_entropy(classe, label)
            model_loss_tot = recon_loss  + self.beta * label_loss
            self.log("val_recon_loss", recon_loss)
        else:
            label_loss = torch.nn.functional.cross_entropy(classe, label)
            model_loss_tot =  label_loss
        self.log("val_loss", model_loss_tot)
        self.log("val_label_loss", label_loss)

        self.recon_to_plot = recon_batch[0][0].cpu()
        self.test_to_plot = volume[0][0].cpu()
        self.label += label.cpu().tolist()
        self.classe  += classe.cpu().tolist()
        return model_loss_tot


    def on_validation_epoch_end(self) -> None:
        print(self.classe)
        print(self.label)
        self.logger.experiment.log_confusion_matrix(self.label, self.classe, epoch=self.current_epoch)
        classe = torch.Tensor(self.classe)
        lab =  torch.Tensor(self.label)
        accuracy= (classe.argmax(dim=1) == lab).sum()/(lab.numel())
        self.label=[]
        self.classe=[]
        self.log("val_accuracy", accuracy.mean())
        self.plot_recon()
        self.plot_test()

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim

    def plot_recon(self):
        path = f"{self.run_name}/recon-{self.current_epoch}.gif"
        save_volume_as_gif(self.recon_to_plot, path)
        self.logger.experiment.log_image(
            path, name="reconstruction", image_format="gif", step=self.current_epoch
        )
        os.remove(path)

    def plot_test(self):
        path = f"{self.run_name}/test-{self.current_epoch}.gif"
        save_volume_as_gif(self.test_to_plot, path)
        self.logger.experiment.log_image(
            path, name="test", image_format="gif", step=self.current_epoch
        )
        os.remove(path)