import torch
import torch.nn as nn
import torchvision.models as models


class mass_detection(nn.Module):
    def __init__(self, config):
        super(mass_detection, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        self.densenet.classifier = nn.Linear(1024, 2)

    def forward(self, data, accu_result = None):
        x = data["image"]

        y_output = self.densenet(x)

        return y_output