"""
Module to store utils function and classes for network
"""

from torch import nn


def init_model(model: nn.Module):
    """Apply common initialization strategy to model
    Not on ViT !

    Args:
        model (nn.Module): Model to initialize
    """
    model.apply(init_weights)


def init_weights(model: nn.Module):
    """Initialize weight for networks using Convolution or Linear layers
      (no transformers) using kaiming / He init

    Args:
        model (nn.Module): Module to apply init strategy
    """
    if isinstance(model, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(model.weight)
        nn.init.constant_(model.bias, 0)
    elif isinstance(model, nn.BatchNorm3d):
        nn.init.constant_(model.weight, 1)
        nn.init.constant_(model.bias, 0)
        if model.running_mean.isnan().any():
            model.running_mean.fill_(0)
        if model.running_var.isnan().any():
            model.running_var.fill_(1)


class KLDivLoss(nn.Module):
    """Returns K-L Divergence loss as proposed by Peng et al. 2021 for brain age predicition
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """

    def __init__(self):
        super().__init__()
        self.loss_func = nn.KLDivLoss(reduction="sum")

    def __call__(self, x, y):
        y += 1e-16
        n = y.shape[0]
        loss = self.loss_func(x, y) / n
        return loss
