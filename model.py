import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureModel(nn.Module):
    def __init__(self):
        super(FeatureModel, self).__init__()

        self.eps = 1e-6

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.norm1 = nn.BatchNorm2d(32, track_running_stats=False)

        self.conv2 = nn.Conv2d(32, 128, 3)
        self.norm2 = nn.BatchNorm2d(128, track_running_stats=False)

        self.sim = nn.CosineSimilarity()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, *inputs):
        image, patch = inputs

        img = self.conv1(image)
        img = F.relu(self.norm1(img))

        img = self.conv2(img)
        img = F.relu(self.norm2(img))

        img = img.view(img.size(0), img.size(1), -1)

        feature = self.conv1(patch)
        feature = F.relu(self.norm1(feature))

        feature = self.conv2(feature)
        feature = F.relu(self.norm2(feature))

        feature = feature.view(feature.size(0), feature.size(1), -1)

        similarity = self.sim(img, feature)
        similarity = F.relu(similarity)

        similarity = (similarity + self.eps).log()

        return similarity.view(similarity.size(0), round(similarity.size(1)**0.5), round(similarity.size(1)**0.5))
