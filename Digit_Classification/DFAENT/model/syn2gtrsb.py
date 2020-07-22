import torch.nn as nn
import torch.nn.functional as F
from model.grad_reverse import grad_reverse

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 144, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(144)
        self.conv3 = nn.Conv2d(144,256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn1_de = nn.BatchNorm2d(144)
        self.bn2_de = nn.BatchNorm2d(96)
        self.bn3_de = nn.BatchNorm2d(3)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(stride=2, kernel_size=2, padding=0, return_indices=True)

    def decode(self, z):
        z = z.view(256,256,4,4)
        x = self.unpool3(z, indices3)
        x = self.relu(self.bn1_de(F.conv_transpose2d(x, weight=self.conv3.weight, stride=1, padding=2)))
        x = self.unpool1(x, indices2)
        x = self.relu(self.bn2_de(F.conv_transpose2d(x, weight=self.conv2.weight, stride=1, padding=1)))
        x = self.unpool2(x, indices1)
        x = self.relu(self.bn3_de(F.conv_transpose2d(x, weight=self.conv1.weight, stride=1, padding=2)))
        return x

    def forward(self, x, is_deconv=False):
        x, indices1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x, indices2 = self.maxpool(self.relu(self.bn2(self.conv2(x))))
        x, indices3 = self.maxpool(self.relu(self.bn3(self.conv3(x))))

        if is_deconv:
            x = self.unpool3(x, indices3)
            x = self.relu(self.bn1_de(F.conv_transpose2d(x, weight=self.conv3.weight, stride=1, padding=2)))
            x = self.unpool1(x, indices2)
            x = self.relu(self.bn2_de(F.conv_transpose2d(x, weight=self.conv2.weight, stride=1, padding=1)))
            x = self.unpool2(x, indices1)
            x = self.relu(self.bn3_de(F.conv_transpose2d(x, weight=self.conv1.weight, stride=1, padding=2)))
        return x

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(6400, 512)
        self.bn1_fc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 43)
        self.relu = nn.ReLU()

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = x.view(x.size(0), 6400)
        x = self.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
