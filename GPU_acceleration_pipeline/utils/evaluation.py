import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_metrics(y_true, y_pred): # Renamed from evaluation_metrics for consistency
    return {
        "rmse" :np.sqrt(mean_squared_error(y_true, y_pred)), # Corrected from np.sqr to np.sqrt
        "mae" : mean_absolute_error(y_true, y_pred),
        "r2" : r2_score(y_true, y_pred)
    }

def compare_models(results, primary_metric="rmse"): # Renamed parameter for clarity
    print(f"Model comparison by {primary_metric.upper()}")
    # Ensure the metric exists and handle potential errors if not
    sorted_models = sorted(results, key=lambda x: x[1].get(primary_metric, float('inf')))
    for model_name, metrics_dict in sorted_models: # Unpack tuple more clearly
        print(f"{model_name}: RMSE={metrics_dict.get('rmse', float('nan')):.4f}, MAE={metrics_dict.get('mae', float('nan')):.4f}, R²={metrics_dict.get('r2', float('nan')):.4f}")