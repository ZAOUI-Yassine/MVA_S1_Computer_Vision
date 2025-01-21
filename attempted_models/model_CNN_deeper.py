import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 500


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1,stride=2) # 64x64x32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2,stride=2) # 32x32x64
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2,stride=2) # 16x16x128
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, padding=2,stride=2) #8x8x256
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=5, padding=2,stride=2) #4x4x512
        self.bn5 = nn.BatchNorm2d(512)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.fc = nn.Linear(512, nclasses)

        # Regularization with Dropout
        self.dropout = nn.Dropout(p=0.3)

        # Activation function
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.dropout(self.activation(self.bn1(self.conv1(x))))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.activation(self.bn5(self.conv5(x)))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


"""
import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 500


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2) # 64x64x32
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2) # 64x64x64
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2) # 32x32x128
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, padding=2) #32x32x256
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=5, padding=2) #16x16x512
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=5, padding=2,stride=2) #16x16x512
        self.bn6 = nn.BatchNorm2d(512)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.fc1 = nn.Linear(512, 256)  # First FC layer with 256 neurons
        self.fc2 = nn.Linear(256, nclasses)  # Final FC layer


        # Regularization with Dropout
        self.dropout = nn.Dropout(p=0.3)

        # Activation function
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(self.activation(self.bn3(self.conv3(x))))
        x = self.activation(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        x = self.activation(self.bn5(self.conv5(x)))
        x = self.activation(self.bn6(self.conv6(x)))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

"""