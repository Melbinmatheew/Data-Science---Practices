# 🏡 House Price Prediction Pipeline

A **modular and production-oriented machine learning pipeline** for predicting house prices. This project is structured into two clear stages:

* **Stage 1:** Model training, hyperparameter tuning, and persistence
* **Stage 2:** Model loading and performance evaluation

---

## 📁 Project Structure

```
house_price_pipeline/
│
├── main_stage1_tune.py         # Entry point: Train, tune, and save best models
├── main_stage2_evaluate.py     # Entry point: Load saved models and evaluate
│
├── config/
│   └── model_configs.py        # Models and hyperparameter grids
│
├── pipeline/
│   ├── preprocess.py           # Preprocessing steps and pipelines
│   └── trainer.py              # Core logic for model training and saving
│
├── registry/
│   └── (model_name).pkl        # Serialized best models saved after Stage 1
│
└── utils/
    └── evaluation.py           # Evaluation metrics and model comparison logic
```

---

## 🧠 Key Features

* **Modular two-stage architecture** for clear separation of training and evaluation
* **Plug-and-play model registry** with versioned `.pkl` files
* **Configurable hyperparameter tuning** via `GridSearchCV`
* **Reusable preprocessing pipelines**
* **Robust evaluation metrics** including R², RMSE, MAE, etc.

---

## 📦 Dataset

The dataset is sourced via `kagglehub`:

```python
import kagglehub

# Automatically fetch the latest version
path = kagglehub.dataset_download("juhibhojani/house-price")
print("Path to dataset files:", path)
```

Make sure you have the necessary Kaggle API credentials set up before running this command.

---

## 🚀 Getting Started

### 1. **Install Requirements**

```bash
pip install -r requirements.txt
```

Make sure the following packages are included:

* `scikit-learn`
* `pandas`
* `numpy`
* `kagglehub`
* `joblib`

### 2. **Download Dataset**

Use the provided script or run the `kagglehub` snippet shown above to fetch the dataset.

---

## 🏗️ Usage

### 🔧 Stage 1: Train + Tune + Save

```bash
python main_stage1_tune.py
```

* Loads raw data
* Applies preprocessing (via `preprocess.py`)
* Performs grid search for each model in `model_configs.py`
* Saves the best model to `registry/`

### 📊 Stage 2: Load + Evaluate

```bash
python main_stage2_evaluate.py
```

* Loads saved models from `registry/`
* Reapplies preprocessing (on test/holdout set)
* Computes metrics using `utils/evaluation.py`

---

## 🛠️ Customization

### ✅ Add a New Model

1. Define the model and its hyperparameter grid in `config/model_configs.py`
2. The tuning logic will automatically incorporate it during `main_stage1_tune.py`

### 📁 Change Preprocessing

Edit or extend the pipeline in `pipeline/preprocess.py`. For example:

* Handle missing values
* Encode categorical variables
* Scale features

---

## 📈 Metrics Used

Implemented in `utils/evaluation.py`:

* **R² Score**
* **Root Mean Squared Error (RMSE)**
* **Mean Absolute Error (MAE)**

Comparative plots or summary tables can be easily added in this module.

---

## ✅ Best Practices Followed

* **Separation of Concerns**: Training, configuration, and evaluation are cleanly separated
* **Versioning**: Models are saved and reused from a centralized registry
* **Config-Driven Design**: Model definitions and search spaces are abstracted in configs
* **Scalability**: Adding new models or datasets is straightforward

---

## 📀 Future Enhancements

* Add cross-validation visualization
* Integrate MLflow for model tracking
* Implement logging and exception handling
* Support CLI arguments for dataset path and mode

---

