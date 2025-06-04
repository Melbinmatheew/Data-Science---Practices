import torch  
from config.model_config import get_models_and_parameters
from pipeline.trainer import train_and_save_model
from pipeline.preprocess import build_preprocessor
import pandas as pd
import traceback # Moved import to top

def main_execution(): # Renamed to avoid conflict if other files have 'main'
    # Check GPU availability
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name()}")

    # Load data
    # Using a placeholder path for generality, user should keep their original path
    data_path = r"D:\Melbin\SELF\House-Price-Prediction\data\final_dataset.csv"
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}. Please check the path.")
        return # Exit if data not found

    X = data.drop(columns=['Amount in rupees'])
    y = data['Amount in rupees']

    print(f"Dataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Build preprocessor to determine input dimensions for PyTorch
    # This preprocessor instance is just for getting input_dim
    temp_preprocessor = build_preprocessor(X.copy()) # Use a copy to avoid any modification to original X
    X_transformed_temp = temp_preprocessor.fit_transform(X.copy()) # Use a copy
    input_dim = X_transformed_temp.shape[1]
    print(f"Input dimension after preprocessing: {input_dim}")

    # Get models with correct input dimension
    models_and_parameters = get_models_and_parameters(input_dim)

    # Train models
    for model_name, config in models_and_parameters.items():
        print(f"\n{'='*50}")
        print(f"Tuning and saving: {model_name}")
        print(f"{'='*50}")
        
        try:
            # Pass copies of X and y to prevent modification across iterations
            path = train_and_save_model(X.copy(), y.copy(), model_name, config["model"], config["params"])
            print(f"✓ Saved best model to {path}")
        except Exception as e:
            print(f"✗ Failed to train {model_name}: {e}")
            traceback.print_exc() # Ensure traceback is printed

    print("\nTraining completed!")

if __name__ == '__main__':
    main_execution()