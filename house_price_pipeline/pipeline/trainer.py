import mlflow
import mlflow.sklearn
import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from pipeline.preprocess import build_preprocessor


def train_and_save_model(X, y, model_name, model, param_grid, output_dir= "registry", scoring = "neg_root_mean_squared_error"):
    """
    Train a model using GridSearchCV and save it to the specified directory.

    Args:
        X (pd.DataFrame): Input features.
        y (pd.Series): Target variable.
        model_name (str): Name of the model.
        model: Scikit-learn model instance.
        param_grid (dict): Hyperparameter grid for GridSearchCV.
        output_dir (str): Directory to save the trained model.
        scoring (str): Scoring metric for GridSearchCV.

    Returns:
        None
    """

    preprocessor = build_preprocessor(X)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    search = GridSearchCV(pipeline, param_grid, cv =5, n_jobs =-1, scoring=scoring, verbose=1)
    search.fit(X,y)

    best_model = search.best_estimator_
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_name}.pkl")
    joblib.dump(best_model, path)

    with mlflow.start_run(run_name=f"Tuned_{model_name}"):
        mlflow.log_params(search.best_params_)
        mlflow.sklearn.log_model(best_model, artifact_path="model")
        mlflow.log_metric("best_score", -search.best_score_)

    return path