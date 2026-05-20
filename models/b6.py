import torch
import torch.nn as nn


class Baseline6(nn.Module):

    def __init__(self, input_size=2048, hidden_size=512, num_classes=8):
        super().__init__()

        # ==================================
        # Frame-level LSTM
        # ==================================
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            # dropout=0.2,
            batch_first=True
        )

        # ==================================
        # Classification Head
        # ==================================
        self.fc = nn.Sequential(

            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Input:
            x shape = (B, P, T, F)

        Where:
            B = Batch size
            P = Number of players
            T = Sequence length (frames)
            F = Feature dimension

        Example:
            (B, 12, 9, 2048)
        """

        B, P, T, F = x.shape

        # ==================================
        # Player Pooling
        # Average features across players
        # ==================================

        # (B, 12, 9, 2048) -> (B, 9, 2048)
        x = torch.mean(x, dim=1)

        # ==================================
        # Temporal Modeling with LSTM
        # ==================================

        # Output shape: (B, 9, hidden_size)
        x, _ = self.lstm(x)

        # Take the last timestep
        # (B, hidden_size)
        x = x[:, -1, :]

        # ==================================
        # Classification
        # ==================================

        return self.fc(x)