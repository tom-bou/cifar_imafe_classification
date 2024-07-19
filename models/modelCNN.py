import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, layer_1, layer_2):
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, self.layer_1, 3, padding=1)
        self.conv2 = nn.Conv2d(self.layer_1, self.layer_2, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(self.layer_2 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.layer_2*8*8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x