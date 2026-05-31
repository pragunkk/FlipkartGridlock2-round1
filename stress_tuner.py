import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import r2_score
import warnings
from model_pipeline import TrafficModel

warnings.filterwarnings('ignore')

def get_hackathon_score(y_true, y_pred):
    return max(0, 100 * r2_score(y_true, y_pred))

# 1. Load and prep data ONCE in memory (super fast)
print("Loading data for automated tuning loop...")
train = pd.read_csv('train.csv').sort_values(by=['day', 'timestamp'])
pipeline_prep = TrafficModel()
df_engineered = pipeline_prep.engineer_features(train)

# Time-Split
split_index = int(len(sorted(df_engineered['day'].unique())) * 0.8)
split_day = sorted(df_engineered['day'].unique())[split_index]

train_df = df_engineered[df_engineered['day'] < split_day]
val_df = df_engineered[df_engineered['day'] >= split_day]

y_train = train_df['demand']
X_train = train_df.drop(['Index', 'demand'], axis=1, errors='ignore')
y_val = val_df['demand']
X_val = val_df.drop(['Index', 'demand'], axis=1, errors='ignore')

def objective(trial):
    """The automated loop that Optuna will run repeatedly."""
    
    # 1. The script guesses new parameters
    params = {
        'depth': trial.suggest_int('depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.1, 1.5)
    }
    
    # 2. Inject parameters and train
    model_pipeline = TrafficModel(**params)
    model_pipeline.train(X_train, y_train, X_val, y_val)
    
    # ==========================================
    # TEST 1: Standard Validation
    # ==========================================
    standard_preds = model_pipeline.predict(X_val)
    score_standard = get_hackathon_score(y_val, standard_preds)
    
    # ==========================================
    # TEST 2: Off-Peak / Rush Hour Slicing
    # ==========================================
    # We will test on Off-Peak since the standard val is entirely off-peak 
    # (or you can adjust this mask to fit your data)
    score_temporal = score_standard # Default fallback
    if 'hour' in X_val.columns:
        peak_mask = X_val['hour'].isin([7, 8, 9, 10, 17, 18, 19, 20])
        y_val_off = y_val[~peak_mask]
        X_val_off = X_val[~peak_mask]
        if len(y_val_off) > 0:
            off_preds = model_pipeline.predict(X_val_off)
            score_temporal = get_hackathon_score(y_val_off, off_preds)

    # ==========================================
    # TEST 3: Feature Amnesia (Missing Data Simulation)
    # ==========================================
    X_val_corrupt = X_val.copy()
    try:
        X_val_corrupt_mapped = model_pipeline._apply_historical_mapping(X_val_corrupt)
        X_val_corrupt_mapped['historical_local_demand'] = model_pipeline.global_median_demand
        amnesia_preds = np.clip(model_pipeline.model.predict(X_val_corrupt_mapped), 0, None)
        score_amnesia = get_hackathon_score(y_val, amnesia_preds)
    except Exception:
        score_amnesia = 0 # Punish the model heavily if it crashes

    # 3. Calculate the objective: The Overall Average
    average_score = np.mean([score_standard, score_temporal, score_amnesia])
    
    return average_score

# --- Start the Automation ---
print("\n--- Starting Automated Stress-Test Tuning ---")
# direction='maximize' tells Optuna we want the highest average score possible
study = optuna.create_study(direction='maximize')

# n_trials=50 means it will run the loop 50 times automatically
study.optimize(objective, n_trials=100)

print("\n==========================================")
print(" TUNING COMPLETE")
print("==========================================")
print(f"Best Average Stress-Test Score: {study.best_value:.2f}")
print("Best Parameters to hardcode into final model:")
for key, value in study.best_params.items():
    print(f"    '{key}': {value},")