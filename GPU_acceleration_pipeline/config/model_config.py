from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from config.pytorch_regressor import TorchRegressorWrapper  # Custom PyTorch model

models_and_parameters = {
    "XGBoost_GPU": {
        "model": XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0),
        "params": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.1],
            "model__max_depth": [3, 6],
        }
    },
    "LightGBM_GPU": {
        "model": LGBMRegressor(device='gpu'),
        "params": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.1],
            "model__num_leaves": [31, 50],
        }
    },
    "PyTorchRegressor": {
        "model": TorchRegressorWrapper(),
        "params": {
            "model__lr": [0.001, 0.01],
            "model__epochs": [50, 100]
        }
    }
}
