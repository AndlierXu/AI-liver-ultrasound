import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from model.modules import TimeDistributed, ConvLSTM, BConvLSTM

class lm_vnet(nn.Module):
    def __init__(self, config):
        super(lm_vnet, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        self.densenet.features.conv0 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.densenet.classifier = nn.Linear(1024*2, 2)

        self.bconvlstm = BConvLSTM(1024,1024,(3,3),2, True, True, False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, data, accu_result = None):
        x = data["lm_frames"]
        wt = data["lesion_size"]

        y = x.contiguous().view(-1, x.size(-3),x.size(-2),x.size(-1))
        y = self.densenet.features(y)
        y = y.contiguous().view(x.size(0), -1, y.size(-3),y.size(-2),y.size(-1))

        layer_output, last_states = self.bconvlstm(y)

        y_out = layer_output[0]

        wt = self.softmax(wt)
        wt = wt.view(wt.size(0), wt.size(1), 1, 1, 1)
        wt = wt.expand_as(y_out)
        y_out = y_out * wt
        y_out = y_out.sum(1)

        y_out = self.pool(y_out)
        y_out = torch.flatten(y_out, 1)
        y_out = self.densenet.classifier(y_out)

        return y_out