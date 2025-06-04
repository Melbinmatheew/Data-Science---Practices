import torch
import xgboost as xgb
import lightgbm as lgb

def check_gpu_support():
    """Check if GPU support is available for all frameworks"""
    print("=== GPU Support Check ===")
    
    # PyTorch GPU check
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    # XGBoost GPU check
    try:
        # Create a simple dataset to test XGBoost GPU
        import numpy as np
        from xgboost import DMatrix
        X_test = np.random.random((100, 10))
        y_test = np.random.random(100)
        dtrain = DMatrix(X_test, label=y_test)
        
        # Test XGBoost with GPU
        xgb_gpu = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0, n_estimators=10)
        xgb_gpu.fit(X_test, y_test)
        print("XGBoost GPU: ✓ Working")
    except Exception as e:
        print(f"XGBoost GPU: ✗ Failed - {e}")
    
    # LightGBM GPU check
    try:
        import numpy as np
        X_test = np.random.random((100, 10))
        y_test = np.random.random(100)
        
        lgb_gpu = lgb.LGBMRegressor(device='gpu', n_estimators=10)
        lgb_gpu.fit(X_test, y_test)
        print("LightGBM GPU: ✓ Working")
    except Exception as e:
        print(f"LightGBM GPU: ✗ Failed - {e}")

# Run the check
check_gpu_support()