import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
linalg = np.linalg

X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)

data = []
data2 = []

for i in range(len(y)):
    if y[i] == 0:
        data.append(X[i])
    else:
        data2.append(X[i])

data = np.asarray(data)
data2 = np.asarray(data2)

# print(data2.shape)
# (2, 1000)
plt.scatter(data2[:,0], data2[:,1], c='green')    
plt.scatter(data[:,0], data[:,1], c='yellow')
plt.show()
#%%
import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 56),
            nn.ReLU(),
            nn.Linear(56, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )


    def forward(self, img):
        x = self.model(img)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 56),
            nn.ReLU(),
            nn.Linear(56, 2),
            nn.ReLU(),
        )

    def forward(self, x):
        img = self.model(x)
        return img

#%%
x_train = data
y_train = data2
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
x_test = data
x_test = torch.FloatTensor(x_test)
y_test = data2
y_test = torch.FloatTensor(y_test)

encoder = Encoder()
decoder = Decoder()
criterion = torch.nn.L1Loss()
op_e = torch.optim.SGD(encoder.parameters(), lr = 0.01)
op_d = torch.optim.SGD(decoder.parameters(), lr = 0.01)

encoder.eval()
decoder.eval()
z = encoder(x_train)
y_pred = decoder(z)
before_train = criterion(y_pred.squeeze(), y_test)
print('Test loss before training' , before_train.item())

#%%
encoder.train()
decoder.train()
epoch = 2000
for epoch in range(epoch):
    op_e.zero_grad()
    op_d.zero_grad()
    # Forward pass
    z = encoder(x_train)
    y_pred = decoder(z)
    # Compute Loss
    loss = criterion(y_pred.squeeze(), y_train)
   
    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    # Backward pass
    loss.backward()
    op_e.step()
    op_d.step()
#%%
encoder.eval()
decoder.eval()
z = encoder(x_train)
y_pred = decoder(z)
y_pred=y_pred.cpu().detach().numpy()
plt.scatter(data2[:,0], data2[:,1], c='green') 
plt.scatter(y_pred[:,0], y_pred[:,1], c='yellow') 
plt.show()
    
    
