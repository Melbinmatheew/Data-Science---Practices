from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


models_and_parameters={
    "LinearRegression" : {
        "model" : LinearRegression(),
        "params" : {
            "model__fit_intercept" :[True, False],
            "model__copy_X": [True, False],
        }
    },
    "DecisionTreeRegresssor" : {
        "model" : DecisionTreeRegressor(),
        "params" :{
            "model__criterion" : ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            "model__splitter" : ["best", "random"],
            "model__max_depth" : [None, 5, 10, 15],
            "model__min_samples_split" : [2, 5, 10],
            "model__min_samples_leaf" : [1, 2, 4, 5],
            "model__max_features" : [None, "auto", "sqrt", "log2"]
        }

    },
    "RandomForestRegressor" :{
        "model" : RandomForestRegressor(),
        "params" : {
            "model__n_estimators" : [10, 50, 100],
            "model__criterion" : ["squared_error", "absolute_error", "poisson"],
            "model__max_depth" : [None, 5, 10, 15],
            "model__min_samples_split" : [2, 5, 10],
            "model__min_samples_leaf" : [1, 2, 4],
            "model__max_features" : [None, "auto", "sqrt", "log2"],
            "model__bootstrap" : [True, False]
        }
    },
    "SVR" :{
        'model' : SVR(),
        "params" : {
            "model__kernel" : ["linear", "poly", "rbf", "sigmoid","precomputed"],
            "model__degree" : [3, 4, 5],
            "model__gamma" : ["scale", "auto"],
            "model__C" : [0.1, 1, 10, 100],
            "model__epsilon" : [0.1, 0.2, 0.5]
        }
    },
    "xgboost" : {
        "model" : XGBRegressor(),
        'params': {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 5, 7],
            "model__min_child_weight": [1, 3, 5],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__gamma": [0, 0.1, 0.2]
        }
    }
}