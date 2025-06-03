import os
import joblib
from util.evaluation import evaluate_metrics, compare_models
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load data
data = pd.read_csv(r"D:\Melbin\SELF\House-Price-Prediction\data\final_dataset.csv")

X = data.drop(columns=['Amount in rupees'])
y = data['Amount in rupees']

model_dir = "registry"
results = []

for file in os.listdir(model_dir):
    if file.endswith(".pkl"):
        model_path = os.path.join(model_dir, file)
        model = joblib.load(model_path)
        y_pred = model.predict(X)
        metrics = evaluate_metrics(y, y_pred)
        model_name = file.replace(".pkl", "")
        results.append((model_name, metrics))

compare_models(results, primary_metric="rmse")
