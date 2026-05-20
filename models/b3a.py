from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class Baseline3A(nn.Module):
    
    def __init__(self, num_classes=9):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT )

        # remove classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # output: (B, 2048, 1, 1)

        self.head = nn.Linear(2048, num_classes)


    def forward(self, x):
        """
        x: (B, 3, 224, 224)
        """

        x = self.backbone(x)   # (B, 2048)
        x = x.view(x.size(0), -1)  # flatten (B, 2048)
        x = self.head(x)       # (B, 9)

        return x