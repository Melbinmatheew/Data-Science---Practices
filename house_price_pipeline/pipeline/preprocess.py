from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd


def build_preprocessor(X: pd.DataFrame) -> Pipeline:
    """
    Build a preprocessing pipeline for the given DataFrame.

    Args:
        X (pd.DataFrame): Input DataFrame containing features.

    Returns:
        Pipeline: A scikit-learn Pipeline object for preprocessing.
    """
    num_col = X.select_dtypes(include= ["float64", "int"]).columns
    cat_col = X.select_dtypes(include=["object", "category"]).columns

    numeric_trnsformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer" , SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_trnsformer, num_col),
        ("cat",categorical_transformer, cat_col )

    ])

    return preprocessor