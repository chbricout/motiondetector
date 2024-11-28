import pytest
import torch

from src.config import IM_SHAPE
from src.network.archi import Encoder, Model
from src.network.cnn_net import CNNEncoder, CNNModel
from src.network.conv5_fc3_net import Conv5FC3Encoder, Conv5FC3Model
from src.network.res_net import ResEncoder, ResModel
from src.network.seres_net import SEResEncoder, SEResModel
from src.network.sfcn_net import SFCNEncoder, SFCNModel
from src.network.vit_net import ViTEncoder, ViTModel


@pytest.mark.parametrize(
    "encoder_to_test",
    [
        (CNNEncoder),
        (Conv5FC3Encoder),
        (ResEncoder),
        (SEResEncoder),
        (SFCNEncoder),
        (ViTEncoder),
    ],
)
def test_encoder_init(encoder_to_test: Encoder):
    net = encoder_to_test(im_shape=IM_SHAPE, dropout_rate=0.5)
    assert hasattr(net, "latent_size")
    assert net.latent_size != None


@pytest.mark.parametrize(
    "model_to_test",
    [(CNNModel), (ResModel), (Conv5FC3Model), (SEResModel), (SFCNModel), (ViTModel)],
)
def test_model_init(model_to_test):
    net = model_to_test(im_shape=IM_SHAPE, num_classes=40, dropout_rate=0.5).cuda()
    dummy = torch.rand(IM_SHAPE).unsqueeze(0).cuda()
    preds = net(dummy)

    assert preds.shape == (1, 40)
    assert torch.isnan(preds).sum() == 0
