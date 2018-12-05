import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]

        conv_index = '22'

        if conv_index.find('22') >= 0:
            self.vgg = nn.Sequential(*modules[:5])
        elif conv_index.find('54') >= 0:
            self.vgg = nn.Sequential(*modules[:35])

        self.vgg.requires_grad = False

    def forward(self, x):
        return self.vgg(x)
