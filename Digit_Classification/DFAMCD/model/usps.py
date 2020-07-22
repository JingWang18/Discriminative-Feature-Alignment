import torch
import torch.nn as nn
import torch.nn.functional as F
from model.grad_reverse import grad_reverse


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(48)
        self.bn1_de = nn.BatchNorm2d(32)
        self.bn2_de = nn.BatchNorm2d(1)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(stride=2, kernel_size=2, dilation=(1,1), return_indices=True)

        # decoding 
        self.de_conv1 = nn.ConvTranspose2d(48,32, kernel_size=5, stride=1)
        self.bn1_de_z = nn.BatchNorm2d(32)
        self.de_conv2 = nn.ConvTranspose2d(32,1, kernel_size=5, stride=1)
        self.bn2_de_z = nn.BatchNorm2d(1)

        self.indices1 = []
        self.indices2 = []

################################################################################        
###### uncomment these lines for adaptation scenario from USPS to MNIST ########
################################################################################            
# =============================================================================
        # residual blocks
        self.conv1_res = nn.Conv2d(48, 48, kernel_size=5, stride=1, padding=2)
        self.conv2_res = nn.Conv2d(48, 48, kernel_size=5, stride=1, padding=2)
        self.bn1_res = nn.BatchNorm2d(48)
        self.bn2_res = nn.BatchNorm2d(48)
        # self.conv3_res = nn.Conv2d(48,48, kernel_size=5, stride=1, padding=2)
        # self.conv4_res = nn.Conv2d(48,48, kernel_size=5, stride=1, padding=2)
        # self.bn3_res = nn.BatchNorm2d(48)
        # self.bn4_res = nn.BatchNorm2d(48)
        # self.conv5_res = nn.Conv2d(48,48, kernel_size=5, stride=1, padding=2)
        # self.conv6_res = nn.Conv2d(48,48, kernel_size=5, stride=1, padding=2)
        # self.bn5_res = nn.BatchNorm2d(48)
        # self.bn6_res = nn.BatchNorm2d(48)  
        # self.conv7_res = nn.Conv2d(48,48, kernel_size=5, stride=1, padding=2)
        # self.conv8_res = nn.Conv2d(48,48, kernel_size=5, stride=1, padding=2)
        # self.bn7_res = nn.BatchNorm2d(48)
        # self.bn8_res = nn.BatchNorm2d(48) 
# =============================================================================
    def decode(self, z):
        z = z.view(128,48,4,4)

        # z = self.unpool1(z, self.indices2)
        # z = self.relu(self.bn1_de_z(self.de_conv1(z)))
        # z = self.unpool2(z, self.indices1)
        # x = self.relu(self.bn2_de_z(self.de_conv2(z)))

        # a = z
        # x = self.relu(self.bn1_res(self.conv1_res(z)))
        # x = self.bn2_res(self.conv2_res(x))
        # x = a + x

        x = self.unpool1(z, self.indices2)
        x = self.relu(self.bn1_de(F.conv_transpose2d(x, weight=self.conv2.weight, stride=1)))
        x = self.unpool1(x, self.indices1)
        x = self.relu(self.bn2_de(F.conv_transpose2d(x, weight=self.conv1.weight, stride=1)))
        return x

    def forward(self, x, is_deconv=False):
        x = torch.mean(x,1).view(x.size()[0],1,x.size()[2],x.size()[3])
        x, indices1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x, indices2 = self.maxpool(self.relu(self.bn2(self.conv2(x))))

        self.indices1 = indices1
        self.indices2 = indices2
        
        if is_deconv:            
################################################################################        
###### uncomment these lines for adaptation scenario from USPS to MNIST ########
################################################################################         
# =============================================================================
            # residual blocks
            # a = x
            # x = self.relu(self.bn1_res(self.conv1_res(x)))
            # x = self.bn2_res(self.conv2_res(x))
            # x = a + x
            # b = x
            # x = self.relu(self.bn3_res(self.conv3_res(x)))
            # x = self.bn4_res(self.conv4_res(x))
            # x = b + x     
            # c = x
            # x = self.relu(self.bn5_res(self.conv5_res(x)))
            # x = self.bn6_res(self.conv6_res(x))
            # x = c + x
            # d = x
            # x = self.relu(self.bn7_res(self.conv7_res(x)))
            # x = self.bn8_res(self.conv8_res(x))
            # x = d + x            
# =============================================================================
            x = self.unpool1(x, indices2)
            x = self.relu(self.bn1_de(F.conv_transpose2d(x, weight=self.conv2.weight, stride=1)))
            x = self.unpool1(x, indices1)
            x = self.relu(self.bn2_de(F.conv_transpose2d(x, weight=self.conv1.weight, stride=1)))
        return x

class Predictor(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(48*4*4, 100)
        self.bn1_fc = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn2_fc = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.relu = nn.ReLU()
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = x.view(x.size(0), 48*4*4)
        x = F.dropout(x, training=self.training, p=self.prob)
        x = self.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = self.relu(self.bn2_fc(self.fc2(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = self.fc3(x)
        return x
