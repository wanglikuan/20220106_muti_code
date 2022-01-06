import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils.model_config import CONFIGS_MTL
from torch.autograd import Variable

import collections


###############################################################
##### Neural Network model in Multi-task Learning Version #####
###############################################################
class MultiLeNetR(nn.Module):
    def __init__(self):
        super(MultiLeNetR, self).__init__()
        # self.conv1 = nn.Conv2d(1, 6, stride=2, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(6)
        # self.conv2 = nn.Conv2d(6, 16, stride=2, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(16)
        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 6, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(6),
            # nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 16, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            # nn.Dropout2d(p=0.5),
            # nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(1),
        )
        # self.conv2_drop = nn.Dropout2d()
        self.bottleneck = nn.Linear(784, 50)

    # def dropout2dwithmask(self, x, mask):
    #     channel_size = x.shape[1]
    #     if mask is None:
    #         mask = Variable(torch.bernoulli(torch.ones(1, channel_size, 1, 1) * 0.5).cuda())
    #     mask = mask.expand(x.shape)
    #     return mask

    # def forward(self, x, mask):
    #     x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     x = self.conv2(x)
    #     mask = self.dropout2dwithmask(x, mask)
    #     if self.training:
    #         x = x * mask
    #     x = F.relu(F.max_pool2d(x, 2))
    #     x = x.view(-1, 320)
    #     x = F.relu(self.fc(x))
    #     return x, mask

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        return x


class MultiLeNetO(nn.Module):
    def __init__(self):
        super(MultiLeNetO, self).__init__()
        self.fc1 = nn.Linear(50, 32)
        self.fc2 = nn.Linear(32, 1)  # 2: num_output (binary classification for each class)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        # return F.log_softmax(x)
        return torch.sigmoid(x)

    # def forward(self, x, mask):
    #     x = F.relu(self.fc1(x))
    #     if mask is None:
    #         mask = Variable(torch.bernoulli(x.data.new(x.data.size()).fill_(0.5)))
    #     if self.training:
    #         x = x * mask
    #     x = self.fc2(x)
    #     return F.sigmoid(x), mask
