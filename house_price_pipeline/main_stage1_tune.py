from config.model_config import models_and_parameters
from pipeline.trainer import train_and_save_model
import pandas as pd

data = pd.read_csv(r"D:\Melbin\SELF\House-Price-Prediction\data\final_dataset.csv")

X = data.drop(columns=['Amount in rupees'])
y = data['Amount in rupees']

for model_name, config in models_and_parameters.items():
    print(f"Tuning and saving: {model_name}")
    path = train_and_save_model(X, y, model_name, config["model"], config["params"])
    print(f"Saved best model to {path}\n")