# models/pytorch_regressor.py
import torch
import torch.nn as nn
from skorch import NeuralNetRegressor

class RegressorModule(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

def TorchRegressorWrapper(input_dim=10):
    return NeuralNetRegressor(
        RegressorModule,
        module__input_dim=input_dim,
        max_epochs=20,
        lr=0.01,
        optimizer=torch.optim.Adam,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
