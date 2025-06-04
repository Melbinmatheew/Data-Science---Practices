import torch
import torch.nn as nn
from skorch import NeuralNetRegressor

class RegressorModule(nn.Module):
    def __init__(
        self,
        input_dim=10, # This is num_features after preprocessing
        hidden_units_1=128,
        hidden_units_2=64,
        dropout_rate=0.2
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_units_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units_1, hidden_units_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units_2, 1)
        )

    # Critical: skorch will pass the features tensor as the first argument.
    # The name 'X' is conventional for features.
    def forward(self, X): 
        return self.network(X.float())

def TorchRegressorWrapper(input_dim=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return NeuralNetRegressor(
        RegressorModule,
        module__input_dim=input_dim, # Passed to RegressorModule.__init__
        criterion=nn.MSELoss, 
        max_epochs=20, # Default, overridden by GridSearchCV
        lr=0.01,       # Default, overridden by GridSearchCV
        optimizer=torch.optim.Adam,
        device=device,
        verbose=1, 
        train_split=None, # Correct for GridSearchCV
        
        # No special parameters like dataset__X_name should be needed here.
        # Skorch's default handling of a DataFrame input X (from the preprocessor)
        # when train_split=None, is to convert X to a tensor and pass it as the
        # first argument to the module's forward method.
        
        iterator_train__shuffle=True,
        iterator_train__num_workers=4 if torch.cuda.is_available() else 0,
        iterator_train__pin_memory=True if torch.cuda.is_available() else False,
    )