import os

import lightning
import torch.nn.functional as F
import torch.optim

from src.training.log import save_volume_as_gif

class ClassifierBase(lightning.LightningModule):
    def training_step(self, batch, batch_idx):
        volume = batch["data"]
        label = batch["label"]
        recon_batch, emb, classe = self.forward(volume)
        # LOSS COMPUTE
        if self.use_decoder:
            recon_loss = F.mse_loss(recon_batch, volume)
            label_loss = F.cross_entropy(classe, label)
            model_loss_tot = recon_loss + self.beta * label_loss
            self.log("train_recon_loss", recon_loss)
        else:
            label_loss = F.cross_entropy(classe, label)
            model_loss_tot = label_loss
        self.log("train_loss", model_loss_tot)
        self.log("train_label_loss", label_loss)

        return model_loss_tot

    def validation_step(self, batch, batch_idx):
        ## VAE TESTING PHASE##
        # INFERENCE
        volume = batch["data"]
        label = batch["label"]
        recon_batch, emb, classe = self.forward(volume)
        # LOSS COMPUTE
        if self.use_decoder:
            recon_loss = F.mse_loss(recon_batch, volume)
            label_loss = F.cross_entropy(classe, label)
            model_loss_tot =  self.beta *recon_loss + label_loss
            self.log("val_recon_loss", recon_loss)
        else:
            label_loss = F.cross_entropy(classe, label)
            model_loss_tot = label_loss
        self.log("val_loss", model_loss_tot)
        self.log("val_label_loss", label_loss)

        self.recon_to_plot = recon_batch[0][0].cpu()
        self.test_to_plot = volume[0][0].cpu()
        self.label += label.cpu().tolist()
        self.classe += classe.cpu().tolist()
        return model_loss_tot

    def on_validation_epoch_end(self) -> None:
        print(self.classe)
        print(self.label)

        self.logger.experiment.log_confusion_matrix(
            self.label, self.classe, epoch=self.current_epoch
        )
        classe = torch.Tensor(self.classe)
        lab = torch.Tensor(self.label)
        if lab.shape[1]==3:
            lab=lab.argmax(dim=1)
        accuracy = (classe.argmax(dim=1) == lab).sum() / (lab.numel())
        self.label = []
        self.classe = []
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
