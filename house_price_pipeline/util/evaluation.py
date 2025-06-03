import numpy as np 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluation_metrics(y_true, y_pred):
    return {
        "rmse" :np.sqr(mean_squared_error(y_true, y_pred)),
        "mae" : mean_absolute_error(y_true, y_pred),
        "r2" : r2_score(y_true, y_pred)
    }

def compare_models( results, primary_metrics = "rmse"):
    print(f"Model comparison by {primary_metrics.upper()}")
    sorted_models = sorted(results, key= lambda x: x[primary_metrics])
    for model_name , metrics in sorted_models:
        print(f"{model_name}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")
        
