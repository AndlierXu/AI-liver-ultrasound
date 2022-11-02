import torch
import torch.nn as nn
import torchvision.models as models

class lm_net(nn.Module):
    def __init__(self, config):
        super(lm_net, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        self.densenet.features.conv0 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.densenet.classifier = nn.Linear(1024, 2)

    def forward(self, data, accu_result = None):
        x = data["liver_lesion_image"]
        y_output = self.densenet(x)

        return y_output