import pandas as pd
import numpy as np
import warnings
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from model_pipeline import TrafficModel

warnings.filterwarnings('ignore')

print("1. Loading datasets...")
train = pd.read_csv('train.csv').sort_values(by=['day', 'timestamp']).reset_index(drop=True)
test = pd.read_csv('test.csv')

print("2. Engineering Clean Features...")
pipeline = TrafficModel()
df_train_engineered = pipeline.engineer_features(train)
df_test_engineered = pipeline.engineer_features(test)

y_train = df_train_engineered['demand']
X_train = df_train_engineered.drop(['Index', 'demand'], axis=1, errors='ignore')
X_test = df_test_engineered.drop(['Index', 'demand'], axis=1, errors='ignore')

print("3. Applying Spatial Clustering...")
X_train, X_test = pipeline.apply_clustering(X_train, X_test)

print("4. Formatting Categorical Data for Ensembling...")
cat_cols = X_train.select_dtypes(include=['object']).columns
for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

cat_features_idx = np.where(X_train.dtypes.apply(lambda x: x.name == 'category'))[0]

print("\n--- INITIATING THE ULTIMATE HEAVYWEIGHT ENSEMBLE ---")

# --- MODEL 1: CATBOOST (The Anchor) ---
print("Training Model 1/3: CatBoost...")
cb_model = CatBoostRegressor(
    iterations=2500,           
    learning_rate=0.07615086613647025,        
    depth=4,                   
    l2_leaf_reg=1.0136657632268968,             
    bagging_temperature=0.5637139679053852,   
    random_strength=0.58442350214666,       
    eval_metric='R2',
    random_seed=42,
    verbose=0 
)
cb_model.fit(X_train, y_train, cat_features=cat_features_idx)
cb_preds = np.clip(cb_model.predict(X_test), a_min=0, a_max=None)

# --- MODEL 2: LIGHTGBM (Optuna Tuned) ---
print("Training Model 2/3: LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.0103426070577093,
    max_depth=7,
    num_leaves=31,
    subsample=0.6451251616482673,
    colsample_bytree=0.9498492689236793,
    reg_alpha=0.45438615068686744,
    reg_lambda=0.2955324049011308,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train, y_train, categorical_feature=[X_train.columns[i] for i in cat_features_idx])
lgb_preds = np.clip(lgb_model.predict(X_test), a_min=0, a_max=None)

# --- MODEL 3: XGBOOST (Optuna Tuned) ---
print("Training Model 3/3: XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=1500,
    learning_rate=0.010239883428871324,
    max_depth=7,
    subsample=0.6631168532609447,
    colsample_bytree=0.8763430763886925,
    min_child_weight=4,
    enable_categorical=True, 
    tree_method='hist',
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_preds = np.clip(xgb_model.predict(X_test), a_min=0, a_max=None)

print("\n--- GENERATING FINAL ENSEMBLE PREDICTIONS ---")
# Weighted Average: CatBoost is still the strongest individual base, so it keeps a slight 40% edge
final_ensemble_preds = (cb_preds * 0.40) + (lgb_preds * 0.30) + (xgb_preds * 0.30)

print("Saving ultimate_submission.csv...")
submission = pd.DataFrame({
    'Index': test['Index'],
    'demand': final_ensemble_preds
})

submission.to_csv('ultimate_submission.csv', index=False)
print("SUCCESS: ultimate_submission.csv is ready for HackerEarth!")