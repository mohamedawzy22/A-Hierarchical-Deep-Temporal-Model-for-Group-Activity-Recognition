import torch
import torch.nn as nn

class Baseline8(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_layers, num_classes):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, num_layers, batch_first=True)

        self.lstm2 = nn.LSTM(hidden_size1 * 2, hidden_size2, num_layers, batch_first=True)

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
        # 🔥 FIX 1: permute + contiguous
        x = x.permute(0, 2, 1, 3).contiguous()

        batch_size, sequence_length, num_players, num_features = x.shape

        # 🔥 FIX 2: ممكن تستخدم reshape أو view بعد contiguous
        x = x.view(batch_size * num_players, sequence_length, num_features)

        x, _ = self.lstm1(x)

        x = x.view(batch_size, num_players, sequence_length, -1)

        team1 = x[:, :6, :, :]
        team2 = x[:, 6:, :, :]
        team1 = torch.max(team1, dim=1)[0]
        team2 = torch.max(team2, dim=1)[0]

        x = torch.cat((team1, team2), dim=2)

        x, _ = self.lstm2(x)

        x = x[:, -1, :]

        return self.fc(x)
