# 🏡 House Price Prediction with MLflow Tracking & GPU Acceleration

This project implements a modular, production-ready pipeline for predicting house prices using a suite of machine learning models. It leverages **MLflow** for experiment tracking and model management, and includes **GPU acceleration** for compatible algorithms (e.g., XGBoost).

## 🚀 Key Features

- **Modular Pipeline**: Separates data ingestion, preprocessing, model training, evaluation, and prediction.
- **MLflow Integration**: Automatically logs metrics, parameters, models, and artifacts.
- **Model Variety**: Supports Decision Tree, Support Vector Regression (SVR), Random Forest, and GPU-accelerated XGBoost.
- **GPU Acceleration**: Utilizes GPU where available to accelerate XGBoost training.
- **Configurable Experiments**: Easy toggling between algorithms and settings via configuration.
- **Reproducibility & Scalability**: All experiments are versioned and trackable via MLflow UI or CLI.

---

## 🗂️ Project Structure

🧪 MLflow Experimentation
Every experiment run with run_experiment.py is automatically tracked using MLflow. This includes:

Model type and hyperparameters

Performance metrics (e.g., MAE, RMSE, R²)

Training and validation plots

Artifact logging (model pickle files, feature importance, etc.)

You can start the MLflow UI to visualize experiments:


RUN>>>
mlflow ui

Then open: http://localhost:5000


🖥️ GPU Acceleration
GPU acceleration is automatically applied to XGBoost if a CUDA-enabled GPU is detected. The pipeline checks for GPU availability and switches the tree_method to gpu_hist.

Ensure proper CUDA and XGBoost-GPU installation:


pip install xgboost --upgrade

# Verify GPU support
python -c "import xgboost as xgb; print(xgb.rabit.get_rank())"

✅ How to Run

Install dependencies
 
pip install -r requirements.txt

Run the ML pipeline

📊 Evaluation Metrics
------------------------->>>

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

Cross-validation scores

Feature importances (for tree-based models)