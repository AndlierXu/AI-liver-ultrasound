import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class seg(nn.Module):
    def __init__(self, config):
        super(seg, self).__init__()
        self.seg = models.segmentation.deeplabv3_resnet101(pretrained=True)

    def forward(self, x):

        logits = self.seg(x)

        return logits['out']


