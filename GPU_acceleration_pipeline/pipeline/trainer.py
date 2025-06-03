from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import os
from pipeline.preprocess import build_preprocessor
import mlflow

def train_and_save_model(X, y, model_name, model, params):
    preprocessor = build_preprocessor(X)

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    grid = GridSearchCV(pipe, params, cv=3, n_jobs=-1, scoring='neg_root_mean_squared_error')
    grid.fit(X, y)

    best_model = grid.best_estimator_
    os.makedirs("registry", exist_ok=True)
    path = f"registry/{model_name}.pkl"
    joblib.dump(grid.best_estimator_, path)

    with mlflow.start_run(run_name=f"Tuned_{model_name}"):
        mlflow.log_params(grid.best_params_)
        mlflow.sklearn.log_model(best_model, artifact_path="model")
        mlflow.log_metric("best_score", -grid.best_score_)


    return path
