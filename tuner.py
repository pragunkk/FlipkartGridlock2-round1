import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import r2_score
import warnings
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from model_pipeline import TrafficModel

warnings.filterwarnings('ignore')

print("1. Loading raw dataset...")
# Make sure this matches where your train.csv is located! 
# (Change to 'dataset/train.csv' if it is inside the folder)
train = pd.read_csv('train.csv') 

# CRITICAL: Teammate's logic to sort by time and prevent leakage
train = train.sort_values(by=['day', 'timestamp'])

print("2. Initializing Feature Pipeline...")
pipeline = TrafficModel()
df_engineered = pipeline.engineer_features(train)

print("3. Performing Chronological Time-Split (80/20)...")
unique_days = sorted(df_engineered['day'].unique())
split_index = int(len(unique_days) * 0.8)
split_day = unique_days[split_index]

train_df = df_engineered[df_engineered['day'] < split_day]
val_df = df_engineered[df_engineered['day'] >= split_day]

y_train = train_df['demand']
X_train = train_df.drop(['Index', 'demand'], axis=1, errors='ignore')

y_val = val_df['demand']
X_val = val_df.drop(['Index', 'demand'], axis=1, errors='ignore')

# Identify categoricals for CatBoost
categorical_features = np.where(X_train.dtypes == object)[0]

def objective(trial):
    """
    Optuna will run this function multiple times, testing different parameters
    each time to maximize the R2 score.
    """
    # 1. Define the search space for parameters
    params = {
        'iterations': trial.suggest_int('iterations', 1500, 3000, step=500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
        'depth': trial.suggest_int('depth', 5, 9),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 8),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 0.8),
        'random_strength': trial.suggest_float('random_strength', 0.0, 0.8),
        'eval_metric': 'R2',
        'random_seed': 42,
        'od_type': 'Iter',
        'od_wait': 50,
        'verbose': 0 # Keep console clean during search
    }
    
    # 2. Replicate the KMeans clustering for this specific trial
    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
    
    X_tr_opt = X_train.copy()
    X_va_opt = X_val.copy()
    
    coords_train = X_tr_opt[['latitude', 'longitude']].fillna(0)
    coords_val = X_va_opt[['latitude', 'longitude']].fillna(0)
    
    X_tr_opt['traffic_zone'] = kmeans.fit_predict(coords_train).astype(str)
    X_va_opt['traffic_zone'] = kmeans.predict(coords_val).astype(str)
    
    # 3. Initialize and train the model with trial parameters
    model = CatBoostRegressor(**params)
    
    model.fit(
        X_tr_opt, y_train, 
        cat_features=categorical_features, 
        eval_set=(X_va_opt, y_val), 
        use_best_model=True
    )
    
    # 4. Predict and evaluate
    preds = model.predict(X_va_opt)
    preds = np.clip(preds, a_min=0, a_max=None)
    
    r2 = r2_score(y_val, preds)
    hackathon_score = max(0, 100 * r2)
    
    return hackathon_score

# --- Run the Optuna Study ---
print("\nStarting Optuna Hyperparameter Search...")
study = optuna.create_study(direction='maximize')

# Set n_trials to 20 or 30. (It will take a while, let it run!)
study.optimize(objective, n_trials=20) 

print("\n" + "="*50)
print(f"BEST HACKATHON SCORE ACHIEVED: {study.best_value:.2f}")
print("BEST PARAMETERS FOUND:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")
print("="*50)
print("ACTION: Copy these parameters into your model_pipeline.py!")