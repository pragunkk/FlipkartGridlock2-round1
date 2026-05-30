import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from model_pipeline import TrafficModel

warnings.filterwarnings('ignore')

def get_score(y_true, y_pred, model_name):
    score = max(0, 100 * r2_score(y_true, y_pred))
    print(f" -> {model_name} Score: {score:.2f}")
    return score

print("1. Standard processing applied (Loading and Engineering)...")
train = pd.read_csv('train.csv').sort_values(by=['day', 'timestamp']).reset_index(drop=True)
pipeline = TrafficModel()
df_engineered = pipeline.engineer_features(train)

split_index = int(len(sorted(df_engineered['day'].unique())) * 0.8)
split_day = sorted(df_engineered['day'].unique())[split_index]

X_train = df_engineered[df_engineered['day'] < split_day].drop(['Index', 'demand'], axis=1, errors='ignore')
y_train = df_engineered[df_engineered['day'] < split_day]['demand']
X_val = df_engineered[df_engineered['day'] >= split_day].drop(['Index', 'demand'], axis=1, errors='ignore')
y_val = df_engineered[df_engineered['day'] >= split_day]['demand']

X_train, X_val = pipeline.apply_clustering(X_train, X_val)
categorical_features = np.where(X_train.dtypes == object)[0]

print("\n--- INITIATING ULTRA-SHALLOW ARCHITECTURE ---")

# Clone 1: The 83.80 Baseline (Heavy L2)
print("Training Clone 1 (Depth 5 - Base)...")
cb_1 = CatBoostRegressor(
    iterations=2500, learning_rate=0.05, depth=5, l2_leaf_reg=6, 
    bagging_temperature=0.6, random_strength=0.8, eval_metric='R2', 
    random_seed=1337, verbose=0 
)
cb_1.fit(X_train, y_train, cat_features=categorical_features, eval_set=(X_val, y_val), early_stopping_rounds=100)
preds_1 = np.clip(cb_1.predict(X_val), 0, None)

# Clone 2: Depth 5 Variant (Extreme L2)
print("Training Clone 2 (Depth 5 - Var)...")
cb_2 = CatBoostRegressor(
    iterations=2500, learning_rate=0.055, depth=5, l2_leaf_reg=7, 
    bagging_temperature=0.7, random_strength=0.9, eval_metric='R2', 
    random_seed=2026, verbose=0 
)
cb_2.fit(X_train, y_train, cat_features=categorical_features, eval_set=(X_val, y_val), early_stopping_rounds=100)
preds_2 = np.clip(cb_2.predict(X_val), 0, None)

# Clone 3: Depth 4 (Maximum Generalization)
print("Training Clone 3 (Depth 4 - Min)...")
cb_3 = CatBoostRegressor(
    iterations=2500, learning_rate=0.06, depth=4, l2_leaf_reg=8, 
    bagging_temperature=0.8, random_strength=1.0, eval_metric='R2', 
    random_seed=42, verbose=0 
)
cb_3.fit(X_train, y_train, cat_features=categorical_features, eval_set=(X_val, y_val), early_stopping_rounds=100)
preds_3 = np.clip(cb_3.predict(X_val), 0, None)

print("\n==================================================")
score_1 = get_score(y_val, preds_1, "Clone 1 (Depth 5 - Base)")
score_2 = get_score(y_val, preds_2, "Clone 2 (Depth 5 - Var) ")
score_3 = get_score(y_val, preds_3, "Clone 3 (Depth 4 - Min) ")
print("==================================================\n")

# Voting weights shifted exclusively to the proven architecture
final_ensemble_preds = (preds_1 * 0.50) + (preds_2 * 0.35) + (preds_3 * 0.15)
final_score = max(0, 100 * r2_score(y_val, final_ensemble_preds))
print(f"** FINAL OPTIMIZED BLEND SCORE: {final_score:.2f} **\n")