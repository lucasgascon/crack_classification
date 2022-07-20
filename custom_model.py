import timm
from torch import nn


class CustomModel(nn.Module):
    def __init__(self):
        self.backbone = timm.create_model(
            'resnet50',
            pretrained=True)
        self.head = nn.Linear(1000, 1)

    def forward(self, x):
        hidden = self.backbone(x)
        pred = self.head(hidden)
        return pred
