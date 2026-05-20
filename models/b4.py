
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class Baseline4(nn.Module):

    def __init__(self, num_classes=8, hidden_dim=512):
        super(Baseline4, self).__init__()

        # =========================
        # ResNet50 Backbone
        # =========================
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # keep everything except FC (includes avgpool)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # freeze first two layers
        for param in resnet.layer1.parameters():
            param.requires_grad = False

        for param in resnet.layer2.parameters():
            param.requires_grad = False

        self.feature_dim = 2048

        # =========================
        # LSTM
        # =========================
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        # =========================
        # Classifier
        # =========================
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        x: (B, 9, C, H, W)
        """

        B, T, C, H, W = x.shape

        # =========================
        # CNN per frame
        # =========================
        x = x.view(B * T, C, H, W)

        features = self.backbone(x)  # (B*T, 2048, 1, 1)

        features = features.view(B * T, self.feature_dim)

        # =========================
        # sequence
        # =========================
        features = features.view(B, T, self.feature_dim)

        # =========================
        # LSTM
        # =========================
        out, _ = self.lstm(features)

        out = out[:, -1, :]

        return self.classifier(out)