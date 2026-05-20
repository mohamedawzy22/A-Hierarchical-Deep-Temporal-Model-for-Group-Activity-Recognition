import torch
import torch.nn as nn

class Baseline3B(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),



            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        x: (batch, 12, 2048)  ← فريم واحد بس
        """

        # pooling over players
        x = torch.mean(x, dim=1)  # x: (batch, 2048)

        return self.classifier(x)