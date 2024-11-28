import pytest
import torch

from src.config import IM_SHAPE
from src.network.archi import Encoder
from src.network.sfcn_net import SFCNEncoder, SFCNModel


def test_encoder_init():
    net = SFCNEncoder(im_shape=IM_SHAPE, dropout_rate=0.5)
    assert hasattr(net, "latent_size")
    assert net.latent_size != None


def test_model_init():
    net = SFCNModel(im_shape=IM_SHAPE, num_classes=40, dropout_rate=0.5).cuda()
    dummy = torch.rand(IM_SHAPE).unsqueeze(0).cuda()
    preds = net(dummy)

    assert preds.shape == (1, 40)
    assert torch.isnan(preds).sum() == 0
