import torch
import torch.nn as nn
import torchvision.models as models


class lmc_net(nn.Module):
    def __init__(self, config):
        super(lmc_net, self).__init__()

        self.densenet = models.densenet121(pretrained=True)
        self.densenet.classifier = nn.Linear(1024, 128)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128+24, 64)
        self.fc2 = nn.Linear(64, 2)
        self.fc_feature = nn.Linear(12,24)

        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax()

    def forward(self, data, accu_result = None):
        i = data["liver_lesion_image"]
        f = data["features"]

        i = self.densenet(i)

        f = self.fc_feature(f)
        x = torch.cat([i, f], dim = 1)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        y_output = self.fc2(x)

        return y_output
