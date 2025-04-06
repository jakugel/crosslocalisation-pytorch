import torch.nn as nn
import torch
from utility_layers import EqualizedLinear


class MappingNetworkLabelled(nn.Module):
    def __init__(self, z_dim, w_dim, num_layers, num_classes, equalized=True, map_leaky=False):
        super().__init__()

        if not map_leaky:
            act = nn.LeakyReLU(map_leaky)
        else:
            act = nn.ReLU()

        if equalized:
            layers = [EqualizedLinear(z_dim + num_classes, w_dim)] + \
                     [act, EqualizedLinear(z_dim, w_dim)] * (num_layers - 1)
        else:
            layers = [nn.Linear(z_dim + num_classes, w_dim)] + \
                     [act, nn.Linear(z_dim, w_dim)] * (num_layers - 1)

        self.mapping = nn.Sequential(*layers)

    def forward(self, x, label):
        x = torch.cat((x, label), axis=-1)
        x = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)  # for PixelNorm
        return self.mapping(x)


class MappingNetworkUnlabelled(nn.Module):
    def __init__(self, z_dim, w_dim, num_layers, equalized=True, map_leaky=False):
        super().__init__()

        if not map_leaky:
            act = nn.LeakyReLU(map_leaky)
        else:
            act = nn.ReLU()

        if equalized:
            layers = [act, EqualizedLinear(z_dim, w_dim)] * num_layers
        else:
            layers = [act, nn.Linear(z_dim, w_dim)] * num_layers

        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        x = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)  # for PixelNorm
        return self.mapping(x)