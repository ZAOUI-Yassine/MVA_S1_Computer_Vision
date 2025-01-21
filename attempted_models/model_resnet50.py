import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class Net(nn.Module):
    def __init__(self, num_classes=500, dropout_rate=0.5):
        super(Net, self).__init__()
        # Load ResNet-50 with pretrained weights
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Replace the fully connected layer to match the number of classes
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        # Add dropout after the fully connected layer
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)  
        return x
