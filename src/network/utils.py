import torch.nn as nn

from src.network.cnn_net import CNNModel
from src.network.conv5_fc3_net import Conv5_FC3Model
from src.network.res_net import ResModel
from src.network.seres_net import SEResModel
from src.network.sfcn_net import SFCNModel
from src.network.vit_net import ViTModel


def init_weights(m):
    if (
        isinstance(m, nn.Linear)
        or isinstance(m, nn.Conv3d)
        or isinstance(m, nn.ConvTranspose3d)
    ):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        if m.running_mean.isnan().any():
            m.running_mean.fill_(0)
        if m.running_var.isnan().any():
            m.running_var.fill_(1)


def parse_model(model):
    model_class = None
    if model == "CNN":
        model_class = CNNModel
    elif model == "RES":
        model_class = ResModel
    elif model == "SFCN":
        model_class = SFCNModel
    elif model == "CONV5_FC3":
        model_class = Conv5_FC3Model
    elif model == "SERES":
        model_class = SEResModel
    elif model == "VIT":
        model_class = ViTModel

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
