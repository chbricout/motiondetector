import pytest
import torch
from src.config import IM_SHAPE
from src.network.archi import Encoder, Model
from src.network.cnn_net import CNNModel, CNNEncoder
from src.network.conv5_fc3_net import Conv5FC3Model, Conv5FC3Encoder
from src.network.res_net import ResModel, ResEncoder
from src.network.seres_net import SEResModel, SEResEncoder
from src.network.sfcn_net import SFCNModel, SFCNEncoder
from src.network.vit_net import ViTModel, ViTEncoder


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
    assert net(dummy).shape == (1, 40)


@pytest.mark.parametrize(
    "model_to_test",
    [(CNNModel), (ResModel), (Conv5FC3Model), (SEResModel), (SFCNModel), (ViTModel)],
)
def test_model_change_num(model_to_test):
    net: Model = model_to_test(im_shape=IM_SHAPE, num_classes=40, dropout_rate=0.5)
    net.classifier.change_output_num(3)
    net = net.cuda()
    dummy = torch.rand(IM_SHAPE).unsqueeze(0).cuda()
    assert net(dummy).shape == (1, 3)


@pytest.mark.parametrize(
    "model_to_test",
    [(CNNModel), (ResModel), (Conv5FC3Model), (SEResModel), (SFCNModel), (ViTModel)],
)
def test_model_mc_dropout(model_to_test):
    net: Model = model_to_test(im_shape=IM_SHAPE, num_classes=40, dropout_rate=0.5)
    net.classifier.change_output_num(3)
    net = net.cuda()
    dummy = torch.rand(IM_SHAPE).unsqueeze(0).cuda()
    assert net(dummy).shape == (1, 3)
