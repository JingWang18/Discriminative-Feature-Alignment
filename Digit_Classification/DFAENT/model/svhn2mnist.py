import torch
import torch.nn as nn
import torch.nn.functional as F
from model.grad_reverse import grad_reverse


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn1_de = nn.BatchNorm2d(64)
        self.bn2_de = nn.BatchNorm2d(64)
        self.bn3_de = nn.BatchNorm2d(3)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=4, stride=2, padding=1)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(stride=2, kernel_size=3, padding=1, return_indices=True)   

        self.conv1_res = nn.Conv2d(128,128, kernel_size=5, stride=1, padding=2)
        self.conv2_res = nn.Conv2d(128,128, kernel_size=5, stride=1, padding=2)
        self.bn1_res = nn.BatchNorm2d(128)
        self.bn2_res = nn.BatchNorm2d(128)

    def decode(self, z):
        z = z.view(256,128,8,8)
        a = z
        x = self.relu(self.bn1_res(self.conv1_res(z)))
        x = self.bn2_res(self.conv2_res(x))
        x = a + x
        x = self.relu(self.bn1_de(F.conv_transpose2d(z, weight=self.conv3.weight, stride=1, padding=2)))
        x = self.unpool1(x, self.indices2)
        x = self.relu(self.bn2_de(F.conv_transpose2d(x, weight=self.conv2.weight, stride=1, padding=2)))
        x = self.unpool2(x, self.indices1)
        x = self.relu(self.bn3_de(F.conv_transpose2d(x, weight=self.conv1.weight, stride=1, padding=2)))
        return x

    def forward(self, x, is_deconv=False):
        x, self.indices1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x, self.indices2 = self.maxpool(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        if is_deconv:
            a = x
            x = self.relu(self.bn1_res(self.conv1_res(x)))
            x = self.bn2_res(self.conv2_res(x))
            x = a + x
            x = self.relu(self.bn1_de(F.conv_transpose2d(x, weight=self.conv3.weight, stride=1, padding=2)))
            x = self.unpool1(x, self.indices2)
            x = self.relu(self.bn2_de(F.conv_transpose2d(x, weight=self.conv2.weight, stride=1, padding=2)))
            x = self.unpool2(x, self.indices1)
            x = self.relu(self.bn3_de(F.conv_transpose2d(x, weight=self.conv1.weight, stride=1, padding=2)))
        return x

class Predictor(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.relu = nn.ReLU()
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = x.view(x.size(0), 8192)
        x = self.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = self.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x
