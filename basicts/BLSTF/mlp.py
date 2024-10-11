import torch
from torch import nn


class GraphMLP(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout=0.15):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x + self.fc2(x)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.15),

            nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True),
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        hidden = self.fc(input_data)  # MLP
        hidden = hidden + input_data  # residual
        return hidden
