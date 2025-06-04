from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# Ensure the path to pytorch_regressor is correct based on your project structure
# If model_config.py is in 'config' directory and pytorch_regressor.py is at the root:
# from ..pytorch_regressor import TorchRegressorWrapper
# If both are in the same directory (e.g. 'config') as per your original import:
from config.pytorch_regressor import TorchRegressorWrapper 
import torch

def get_models_and_parameters(input_dim):
    """Get models with proper input dimension for PyTorch"""
    
    models_and_parameters_dict = {
        "XGBoost_GPU": {
            "model": XGBRegressor(
                device='cuda',
                random_state=42,
                # Forcing gpu_hist if 'device=cuda' doesn't automatically use it effectively
                # tree_method='gpu_hist', 
            ),
            "params": {
                "model__n_estimators": [100, 200, 300], # Added 300
                "model__learning_rate": [0.01, 0.05, 0.1], # Added 0.05
                "model__max_depth": [3, 5, 7], # Changed 6 to 5, 7
            }
        },
        "LightGBM_GPU": {
            "model": LGBMRegressor(
                device='gpu',
                random_state=42,
                verbose=-1
            ),
            "params": {
                "model__n_estimators": [100, 200, 300], # Added 300
                "model__learning_rate": [0.01, 0.05, 0.1], # Added 0.05
                "model__num_leaves": [31, 50, 70], # Added 70
            }
        },
        # "PyTorchRegressor": {
        #     "model": TorchRegressorWrapper(input_dim=input_dim),
        #     "params": {
        #         "model__lr": [0.001, 0.005, 0.01], # Added 0.005
        #         "model__max_epochs": [50, 100, 150], # Expanded epochs
        #         "model__batch_size": [128, 256, 512], # Tunable batch_size
        #         "model__optimizer__weight_decay": [0, 1e-5, 1e-4], # Added L2 regularization
        #         # Parameters for RegressorModule (prefixed with 'model__module__')
        #         "model__module__hidden_units_1": [64, 128], # Neurons in first hidden layer
        #         "model__module__hidden_units_2": [32, 64],   # Neurons in second hidden layer
        #         "model__module__dropout_rate": [0.1, 0.2, 0.3] # Dropout rate
        #     }
        # }
    }
    
    return models_and_parameters_dict

# For backward compatibility or direct use (less flexible for input_dim)
# This global models_and_parameters will be overwritten if main_stage1_tune.py calls get_models_and_parameters.
# It's generally better to rely on the get_models_and_parameters function.
models_and_parameters = {
    "XGBoost_GPU": {
        "model": XGBRegressor(
            device='cuda',
            random_state=42
        ),
        "params": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.1],
            "model__max_depth": [3, 6],
        }
    },
    "LightGBM_GPU": {
        "model": LGBMRegressor(
            device='gpu',
            random_state=42,
            verbose=-1
        ),
        "params": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.1],
            "model__num_leaves": [31, 50],
        }
    },
#     "PyTorchRegressor": {
#         "model": TorchRegressorWrapper(), # This won't get dynamic input_dim from main script
#         "params": {
#             "model__lr": [0.001, 0.01],
#             "model__max_epochs": [50, 100],
#             "model__batch_size": [128, 256],
#             # Add other new params here if using this global dict directly
#         }
#     }
}