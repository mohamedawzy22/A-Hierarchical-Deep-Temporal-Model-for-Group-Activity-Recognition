import torch
import torch.nn as nn


class Baseline8(nn.Module):
    """
    Baseline 8:
    -------------------------
    Hierarchical dual-LSTM model for group activity recognition.

    Idea:
    - Each player sequence is processed independently using the first LSTM
      to learn player motion dynamics over time.
    - Players are then divided into two teams (6 vs 6).
    - Team-wise max pooling is applied to summarize each team representation.
    - The pooled team representations are concatenated together.
    - A second LSTM learns temporal interaction between both teams.
    - Final fully connected layers perform group activity classification.

    Pipeline:
    Player crops
        ↓
    ResNet embeddings
        ↓
    Player-level temporal modeling (LSTM 1)
        ↓
    Team-wise pooling
        ↓
    Team interaction temporal modeling (LSTM 2)
        ↓
    Group activity classification
    """

    def __init__(self, input_size, hidden_size1, hidden_size2,
                 num_layers, num_classes):

        super().__init__()

        # First LSTM:
        # learns temporal dynamics for each player independently
        self.lstm1 = nn.LSTM(
            input_size,
            hidden_size1,
            num_layers,
            batch_first=True
        )

        # Second LSTM:
        # learns temporal interaction between the two teams
        self.lstm2 = nn.LSTM(
            hidden_size1 * 2,
            hidden_size2,
            num_layers,
            batch_first=True
        )

        # Final classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):

        # Input shape:
        # [batch_size, num_players, sequence_length, feature_dim]

        # Rearrange dimensions to:
        # [batch_size, sequence_length, num_players, feature_dim]
        x = x.permute(0, 2, 1, 3).contiguous()

        batch_size, sequence_length, num_players, num_features = x.shape

        # Merge batch and player dimensions
        # so each player sequence is processed independently
        x = x.view(
            batch_size * num_players,
            sequence_length,
            num_features
        )

        # Player-level temporal modeling
        x, _ = self.lstm1(x)

        # Restore player dimension
        x = x.view(
            batch_size,
            num_players,
            sequence_length,
            -1
        )

        # Split players into two teams
        team1 = x[:, :6, :, :]
        team2 = x[:, 6:, :, :]

        # Team-wise max pooling
        # summarizes players inside each team
        team1 = torch.max(team1, dim=1)[0]
        team2 = torch.max(team2, dim=1)[0]

        # Concatenate both team representations
        x = torch.cat((team1, team2), dim=2)

        # Higher-level temporal modeling between teams
        x, _ = self.lstm2(x)

        # Use final temporal state for classification
        x = x[:, -1, :]

        # Final group activity prediction
        return self.fc(x)