import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

# Import the model your teammate built (Contract remains unchanged)
from model_pipeline import TrafficModel

warnings.filterwarnings('ignore')

def calculate_metrics(y_true, y_pred, fold_name, print_results=True):
    """Calculates standard ML metrics + the specific Hackathon metric."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    hackathon_score = max(0, 100 * r2)
    
    if print_results:
        print(f"\n{'='*50}")
        print(f" {fold_name} METRICS")
        print(f"{'='*50}")
        print(f" Hackathon Score (100*R2) : {hackathon_score:.2f}")
        print(f" RMSE                     : {rmse:.4f}")
        print(f" MAE                      : {mae:.4f}")
        print(f" Raw R2                   : {r2:.4f}")
        print(f"{'='*50}\n")
        
    return hackathon_score, rmse

def run_evaluation():
    print("1. Loading raw dataset...")
    train = pd.read_csv('train.csv')
    train = train.sort_values(by=['day', 'timestamp'])
    
    print("2. Initializing Teammate's Model and processing features...")
    pipeline = TrafficModel()
    df_engineered = pipeline.engineer_features(train)
    
    print("3. Performing Chronological Time-Split...")
    unique_days = sorted(df_engineered['day'].unique())
    split_index = int(len(unique_days) * 0.8)
    split_day = unique_days[split_index]
    
    train_df = df_engineered[df_engineered['day'] < split_day]
    val_df = df_engineered[df_engineered['day'] >= split_day]
    
    y_train = train_df['demand']
    X_train = train_df.drop(['Index', 'demand'], axis=1, errors='ignore')
    
    y_val = val_df['demand']
    X_val = val_df.drop(['Index', 'demand'], axis=1, errors='ignore')
    
    print("\n4. Training Model...")
    pipeline.train(X_train, y_train, X_val, y_val)
    
    print("\n5. Running Inference (Standard Local Test)...")
    val_predictions = pipeline.predict(X_val)
    val_predictions = np.clip(val_predictions, a_min=0, a_max=None)
    calculate_metrics(y_val, val_predictions, "STANDARD CHRONOLOGICAL VALIDATION")
    
    # ---------------------------------------------------------
    # ADVANCED TEST 1: RUSH HOUR SLICING
    # ---------------------------------------------------------
    print("6. Running Stress Test: Peak vs Off-Peak Hours...")
    # Assuming 'hour' is available from the engineered features
    if 'hour' in X_val.columns:
        # Define peak hours (e.g., 7-10 AM, 5-8 PM)
        peak_mask = X_val['hour'].isin([7, 8, 9, 10, 17, 18, 19, 20])
        
        y_val_peak = y_val[peak_mask]
        preds_peak = val_predictions[peak_mask]
        
        y_val_off = y_val[~peak_mask]
        preds_off = val_predictions[~peak_mask]
        
        if len(y_val_peak) > 0:
            calculate_metrics(y_val_peak, preds_peak, "STRESS TEST: RUSH HOUR TRAFFIC ONLY")
        if len(y_val_off) > 0:
            calculate_metrics(y_val_off, preds_off, "STRESS TEST: OFF-PEAK TRAFFIC ONLY")
            
    # ---------------------------------------------------------
    # ADVANCED TEST 2: MISSING SENSOR DATA INJECTION (FIXED)
    # ---------------------------------------------------------
    print("7. Running Stress Test: Missing Feature Robustness (15% Data Drop)...")
    
    # 1. Grab the RAW validation data (before any engineering)
    raw_val_df = train[train['day'] >= split_day].copy()
    
    # 2. Inject the missing data into the raw source
    for col in ['Weather', 'Temperature']:
        if col in raw_val_df.columns:
            mask = np.random.rand(len(raw_val_df)) < 0.15
            raw_val_df.loc[mask, col] = np.nan
            
    try:
        # 3. Force the pipeline to engineer features on the corrupted data
        corrupted_engineered = pipeline.engineer_features(raw_val_df)
        X_val_corrupted = corrupted_engineered.drop(['Index', 'demand'], axis=1, errors='ignore')
        
        # 4. Predict and evaluate
        corrupted_preds = pipeline.predict(X_val_corrupted)
        calculate_metrics(y_val, corrupted_preds, "STRESS TEST: 15% MISSING SENSOR DATA")
    except Exception as e:
        print(f"\n[CRITICAL FAILURE] Pipeline crashed on missing data!\nError: {e}")

if __name__ == "__main__":
    run_evaluation()