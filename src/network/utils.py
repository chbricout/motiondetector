"""
Module to store utils function and classes for network
"""

import logging
from typing import Type
from torch import nn

from src.network.archi import Model
from src.network.cnn_net import CNNModel
from src.network.conv5_fc3_net import Conv5FC3Model
from src.network.res_net import ResModel
from src.network.seres_net import SEResModel
from src.network.sfcn_net import SFCNModel
from src.network.vit_net import ViTModel, ViTClassifier, ViTEncoder


def init_model(model: nn.Module):
    """Apply common initialization strategy to model
    Not on ViT !

    Args:
        model (nn.Module): Model to initialize
    """
    if not isinstance(model, (ViTModel, ViTClassifier, ViTEncoder)):
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


def parse_model(model: str) -> Type[Model]:
    """Return the class corresponding to a model string name

    Args:
        model (str): model name (capitalized)

    Returns:
        Model: model class
    """
    model_class: Model = None
    if model == "CNN":
        model_class = CNNModel
    elif model == "RES":
        model_class = ResModel
    elif model == "SFCN":
        model_class = SFCNModel
    elif model == "CONV5_FC3":
        model_class = Conv5FC3Model
    elif model == "SERES":
        model_class = SEResModel
    elif model == "VIT":
        model_class = ViTModel
    else:
        logging.error("No model corresponding to %s, use CNNModel by default", model)
        model_class = CNNModel

    return model_class


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
