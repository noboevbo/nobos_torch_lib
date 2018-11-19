import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.distributed
import torch.utils.data.distributed
from nobos_commons.data_structures.dimension import ImageSize


class ColorActionNetV1(nn.Module):
    def __init__(self, heatmap_size: ImageSize, num_classes: int, num_joints: int, num_channels: int):
        super(ColorActionNetV1, self).__init__()
        self.heatmap_size = heatmap_size
        self.fc1_size = int(0.5 * heatmap_size.width * heatmap_size.height)
        self.fc1_size = 57344

        self.conv1 = nn.Conv2d(num_joints * num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.global_avg = nn.AvgPool2d(512)
        self.fc1 = torch.nn.Linear(self.fc1_size, num_classes)

    def forward(self, x):
        # Example sizes:
        # Input 1, 57, 64, 114
        # conv1: 1, 32, 64, 114
        # pool1: 1, 32, 32, 57
        # conv2: 1, 128, 32, 57
        # pool2: 1, 32, 12, 12
        # conv3: 1, 64, 12, 12
        # conv4: 1, 32, 12, 12
        # pool3: 1, 32, 6, 6
        width = x.shape[2]
        height = x.shape[3]
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        x = x.view(-1, self.fc1_size) # 0.5*w*h
        x = self.fc1(x)
        return x