import torch  
import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import os
from pipeline.preprocess import build_preprocessor # Assuming this path is correct
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd # Ensure pandas is imported if used for type checking
import numpy as np # Ensure numpy is imported for y.values

def train_and_save_model(X_orig, y_orig, model_name, model, params): # Use X_orig, y_orig
    print(f"Starting training for {model_name}")

    # Create copies to modify for specific models
    X = X_orig.copy()
    y = y_orig.copy()
    
    # Monitor GPU usage if available
    if torch.cuda.is_available():
        # Note: torch.cuda.memory_allocated() only tracks memory allocated by PyTorch.
        # XGBoost and LightGBM manage their own GPU memory.
        print(f"PyTorch GPU Memory before training {model_name}: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    
    # Convert data types based on model
    # This conversion happens *before* preprocessing.
    # The preprocessor (e.g., StandardScaler) might output float64.
    # PyTorch module's forward pass has .float() to handle this.
    if model_name == "PyTorchRegressor":
        # Ensure X's numeric columns are float32.
        # If X contains non-numeric columns that shouldn't be cast, select them carefully.
        # Assuming X passed here is the raw DataFrame.
        for col in X.select_dtypes(include=np.number).columns:
            X[col] = X[col].astype('float32')
        y = y.values.astype('float32').reshape(-1, 1)  # Ensure proper shape for PyTorch
    
    # Preprocessor is built using the potentially type-casted X for PyTorch,
    # or original X for other models (due to X.copy() at start of function).
    preprocessor = build_preprocessor(X)
    
    # Define the full pipeline
    # Note: set_output(transform="pandas") on pipeline is fine.
    # Skorch will convert pandas DataFrame from preprocessor to numpy then to tensor.
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    # pipe.set_output(transform="pandas") # This was on XGBoost, let's be consistent if needed, or remove if not
    # Actually, set_output on the pipeline itself is better if the final estimator also supports pandas input
    # or if we want pandas output from predict/transform.
    # For training with scikit-learn GridSearchCV, it expects NumPy arrays or DataFrames.
    # Skorch handles DataFrame input. XGBoost/LightGBM also handle DataFrame input.

    print(f"Running GridSearchCV with parameter combinations for {model_name}")
    
    grid = GridSearchCV(
        pipe, 
        params, 
        cv=3, 
        n_jobs=1,
        scoring='neg_root_mean_squared_error',
        verbose=1
    )

    start_time = time.time()
    grid.fit(X, y)
    end_time = time.time()

    best_model = grid.best_estimator_
    os.makedirs("registry", exist_ok=True)
    local_path = f"registry/{model_name}.pkl"
    joblib.dump(best_model, local_path)

    # === MLflow Logging & Registration ===
    with mlflow.start_run(run_name=f"Tuned_{model_name}") as run:
        run_id = run.info.run_id

        # 1. Log parameters and metrics
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("best_score_rmse", -grid.best_score_)
        mlflow.log_metric("training_time_seconds", end_time - start_time)
        mlflow.set_tag("model_name", model_name)

        # 2. Log model
        mlflow.sklearn.log_model(sk_model=best_model, artifact_path="model")

        # 3. Register model
        model_uri = f"runs:/{run_id}/model"
        registered_model_name = f"{model_name}_regressor"

        client = MlflowClient()
        try:
            model_details = client.register_model(model_uri=model_uri, name=registered_model_name)
            print(f"✓ Model registered: {registered_model_name} (version {model_details.version})")
        except Exception as e:
            print(f"✗ Failed to register model: {e}")

    print(f"Best parameters for {model_name}: {grid.best_params_}")
    print(f"Best RMSE for {model_name}: {-grid.best_score_:.4f}")
    
    return local_path