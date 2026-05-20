from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class Baseline1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # =========================
        # Freeze layer1 + layer2
        # =========================
        for param in resnet.layer1.parameters():
            param.requires_grad = False

        for param in resnet.layer2.parameters():
            param.requires_grad = False

    
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        num_features = resnet.fc.in_features
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x