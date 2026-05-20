import torch
import torch.nn as nn


class Baseline5(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()

        # ==================================
        # LSTM (Player-level temporal model)
        # ==================================
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # ==================================
        # Classifier
        # ==================================
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        x: (B, 12, 9, 2048)
        """

        B, P, T, F = x.shape

        # ==================================
        # Flatten players into batch
        # ==================================
        x = x.view(B * P, T, F)  # (B*12, 9, 2048)

        # ==================================
        # LSTM
        # ==================================
        out, _ = self.lstm(x)  # (B*P, T, H)

        # Take last timestep
        player_feats = out[:, -1, :]  # (B*P, H)

        # ==================================
        # Restore player dimension
        # ==================================
        player_feats = player_feats.view(B, P, -1)  # (B, 12, H)

        # ==================================
        # Pool over players
        # ==================================
        group_feat,_ = torch.max(player_feats, dim=1)  # (B, H)

        # ==================================
        # Classification
        # ==================================
        out = self.classifier(group_feat)

        return out