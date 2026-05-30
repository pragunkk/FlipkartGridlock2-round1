import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb
import warnings
from model_pipeline import TrafficModel

warnings.filterwarnings('ignore')

print("1. Preparing Data for Heavyweight Tuning...")
train = pd.read_csv('train.csv').sort_values(by=['day', 'timestamp']).reset_index(drop=True)

pipeline = TrafficModel()
df_engineered = pipeline.engineer_features(train)

# 80/20 Time Split
split_index = int(len(sorted(df_engineered['day'].unique())) * 0.8)
split_day = sorted(df_engineered['day'].unique())[split_index]

X_train_full = df_engineered[df_engineered['day'] < split_day].drop(['Index', 'demand'], axis=1, errors='ignore')
y_train = df_engineered[df_engineered['day'] < split_day]['demand']
X_val_full = df_engineered[df_engineered['day'] >= split_day].drop(['Index', 'demand'], axis=1, errors='ignore')
y_val = df_engineered[df_engineered['day'] >= split_day]['demand']

X_train, X_val = pipeline.apply_clustering(X_train_full, X_val_full)

# Format for LightGBM/XGBoost
cat_cols = X_train.select_dtypes(include=['object']).columns
for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_val[col] = X_val[col].astype('category')
cat_features_idx = np.where(X_train.dtypes.apply(lambda x: x.name == 'category'))[0]

# ==========================================
# OPTUNA STUDY 1: LIGHTGBM
# ==========================================
def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000, step=500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255, step=16),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': 42,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, categorical_feature=[X_train.columns[i] for i in cat_features_idx])
    preds = np.clip(model.predict(X_val), 0, None)
    return max(0, 100 * r2_score(y_val, preds))

# ==========================================
# OPTUNA STUDY 2: XGBOOST
# ==========================================
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 2500, step=500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'enable_categorical': True,
        'tree_method': 'hist',
        'random_state': 42
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_val), 0, None)
    return max(0, 100 * r2_score(y_val, preds))

print("\n--- STARTING LIGHTGBM OPTIMIZATION ---")
study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=15) # Adjust n_trials based on your time

print("\n--- STARTING XGBOOST OPTIMIZATION ---")
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=15) 

print("\n" + "="*50)
print(f"BEST LIGHTGBM SCORE: {study_lgb.best_value:.2f}")
print("BEST LIGHTGBM PARAMS:")
for key, value in study_lgb.best_params.items():
    print(f"    {key}: {value}")

print("-" * 50)
print(f"BEST XGBOOST SCORE: {study_xgb.best_value:.2f}")
print("BEST XGBOOST PARAMS:")
for key, value in study_xgb.best_params.items():
    print(f"    {key}: {value}")
print("="*50)