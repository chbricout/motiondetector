import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        if m.running_mean.isnan().any():
            m.running_mean.fill_(0)
        if m.running_var.isnan().any():
            m.running_var.fill_(1)


