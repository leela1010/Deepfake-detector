import torch
import torch.nn as nn
import torchvision.models as models

class DeepfakeModel(nn.Module):
    def __init__(self):
        super(DeepfakeModel, self).__init__()
        # Load ResNet18 without pre-trained weights (we’ll load DFDC weights separately)
        self.model = models.resnet18(weights=None)
        # Replace the final fully connected layer with 1 output for binary classification
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)
