import torch
from torch import nn


class Baseline7(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size1,
        hidden_size2,
        num_layers,
        num_classes
    ):
        super().__init__()

        self.hidden_size1 = hidden_size1

        # ==================================
        # LSTM 1
        # Player-level temporal modeling
        # ==================================
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size1,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        # ==================================
        # LSTM 2
        # Frame-level temporal modeling
        # ==================================
        self.lstm2 = nn.LSTM(
            input_size=hidden_size1,
            hidden_size=hidden_size2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        # ==================================
        # Classification Head
        # ==================================
        self.fc = nn.Sequential(
            nn.Linear(hidden_size2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Input:
            x: (B, P, T, F)
        """

        B, P, T, F = x.shape
        H1 = self.hidden_size1

        # ==================================
        # LSTM1: Player-level temporal modeling
        # ==================================
        x = x.view(B * P, T, F)
        x, _ = self.lstm1(x)

        # Restore shape
        x = x.view(B, P, T, H1)

        # ==================================
        # Pool over players (MEAN)
        # ==================================

        x = x.permute(0, 2, 1, 3).contiguous()  # (B, T, P, H1)

        x = x.view(B * T, P, H1)

        x, _ = torch.max(x, dim=1)  # (B*T, H1)

        x = x.view(B, T, H1)

        # ==================================
        # LSTM2: Frame-level temporal modeling
        # ==================================
        x, _ = self.lstm2(x)

        x = x[:, -1, :]

        # ==================================
        # Classification
        # ==================================
        return self.fc(x)