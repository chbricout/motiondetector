import os

import lightning
import torch.nn.functional as F
import torch.optim
import torchmetrics
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import balanced_accuracy_score

from src.training.log import save_volume_as_gif

class ReconstructBase(lightning.LightningModule):

  

    def training_step(self, batch, batch_idx):
        volume = batch["data"]
        label = batch["label"]
        recon_batch, emb, classe = self.forward(volume)
        # LOSS COMPUTE
        if self.use_decoder:
            recon_loss = self.recon_loss(recon_batch, volume)
            label_loss = self.label_loss(classe, label)
            model_loss_tot = recon_loss + self.beta * label_loss
            self.log("train_recon_loss", recon_loss)
        else:
            label_loss = self.label_loss(classe, label)
            model_loss_tot = label_loss
        self.log("train_loss", model_loss_tot)
        self.log("train_label_loss", label_loss)

        self.plot_train(volume[0][0].cpu())
        return model_loss_tot

    def validation_step(self, batch, batch_idx):
        ## VAE TESTING PHASE##
        # INFERENCE
        volume = batch["data"]
        label = batch["label"]
        recon_batch, emb, classe = self.forward(volume)
        # LOSS COMPUTE
        if self.use_decoder:
            recon_loss = self.recon_loss(recon_batch, volume)
            label_loss = self.label_loss(classe, label)
            model_loss_tot =  self.beta *recon_loss + label_loss
            self.log("val_recon_loss", recon_loss)
        else:
            label_loss = self.label_loss(classe, label)
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
        classe = torch.Tensor(self.classe)
        lab = torch.Tensor(self.label)
        if len(lab.shape)==2:
            lab=lab.argmax(dim=1)
        if lab.dtype == torch.float32:
            lab =lab.int()


        if len(classe.shape)==2:
            classe=classe.argmax(dim=1)    
        if classe.dtype == torch.float32:
            classe =classe.round().int()

        accuracy = (classe == lab).sum() / (lab.numel())
        self.logger.experiment.log_confusion_matrix(
            lab, classe, epoch=self.current_epoch
        )
        self.label = []
        self.classe = []
        self.log("val_accuracy", accuracy.mean())
        self.log("val_balanced_accuracy", balanced_accuracy_score(lab,classe))
        self.plot_recon()
        self.plot_test()

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = StepLR(optim, 40, 0.8)
        return [optim], [{"scheduler": scheduler, "interval": "epoch"}]
    
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
        
    def plot_train(self, train_to_plot):
        path = f"{self.run_name}/train-{self.current_epoch}-{self.global_step}.gif"
        save_volume_as_gif(train_to_plot, path)
        self.logger.experiment.log_image(
            path, name="train", image_format="gif", step=self.global_step
        )
        os.remove(path)


class ClassifierBase(lightning.LightningModule):

    def training_step(self, batch, batch_idx):
        volume = batch["data"]
        label = batch["label"]
        emb, classe = self.forward(volume)
        # LOSS COMPUTE
       
        label_loss = self.label_loss(classe, label)
        model_loss_tot = label_loss
        self.log("train_loss", model_loss_tot)
        self.log("train_label_loss", label_loss)

        self.plot_train(volume[0][0].cpu())
        return model_loss_tot

    def validation_step(self, batch, batch_idx):
        ## VAE TESTING PHASE##
        # INFERENCE
        volume = batch["data"]
        label = batch["label"]
        emb, classe = self.forward(volume)
        # LOSS COMPUTE
       
        label_loss = self.label_loss(classe, label)
        model_loss_tot = label_loss
        self.log("val_loss", model_loss_tot)
        self.log("val_label_loss", label_loss)

        self.test_to_plot = volume[0][0].cpu()
        self.label += label.cpu().tolist()
        self.classe += classe.sigmoid().cpu().tolist()
        return model_loss_tot

    def on_validation_epoch_end(self) -> None:
        print(self.classe)
        print(self.label)
        classe = torch.Tensor(self.classe)
        lab = torch.Tensor(self.label)
        if len(lab.shape)==2:
            lab=lab.argmax(dim=1)
        if lab.dtype == torch.float32:
            lab =lab.int()


        if self.mode=="CLASS":
            classe=classe.round().int()
        elif classe.dtype == torch.float32:
            classe =classe.round().int()

        accuracy = (classe == lab).sum() / (lab.numel())
        self.logger.experiment.log_confusion_matrix(
            lab, classe, epoch=self.current_epoch
        )
        self.label = []
        self.classe = []
        self.log("val_accuracy", accuracy.mean())
        self.log("val_balanced_accuracy", balanced_accuracy_score(lab,classe))
        self.plot_test()

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = StepLR(optim, 500, 0.8)
        return [optim], [{"scheduler": scheduler, "interval": "epoch"}]
    

    def plot_test(self):
        path = f"{self.run_name}/test-{self.current_epoch}.gif"
        save_volume_as_gif(self.test_to_plot, path)
        self.logger.experiment.log_image(
            path, name="test", image_format="gif", step=self.current_epoch
        )
        os.remove(path)
        
    def plot_train(self, train_to_plot):
        path = f"{self.run_name}/train-{self.current_epoch}-{self.global_step}.gif"
        save_volume_as_gif(train_to_plot, path)
        self.logger.experiment.log_image(
            path, name="train", image_format="gif", step=self.global_step
        )
        os.remove(path)
