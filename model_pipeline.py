import pandas as pd
import numpy as np
import pygeohash as pgh
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

class TrafficModel:
    def __init__(self, **kwargs):
        # Set our reliable baselines
        default_params = {
            'iterations': 1500,           
            'learning_rate': 0.07615086613647025,        
            'depth': 4,                   
            'eval_metric': 'R2',
            'random_seed': 42,
            'od_type': 'Iter',
            'od_wait': 50,       
            'l2_leaf_reg': 1.0136657632268968,
            'bagging_temperature': 0.5637139679053852,
            'random_strength': 0.58442350214666,        
            'verbose': 0  # Turn off printing so it doesn't spam the console during tuning
        }
        
        # Override baselines with whatever Optuna suggests
        default_params.update(kwargs)
        
        self.model = CatBoostRegressor(**default_params)
        self.kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
        self.is_kmeans_fitted = False 
        self.historical_mapping = None
        self.global_median_demand = 0

    def engineer_features(self, df):
        """Transforms raw data into pure, clean machine-readable math."""
        df_engineered = df.copy()
        
        # 1. Spatial Features
        if 'geohash' in df_engineered.columns:
            df_engineered['latitude'] = df_engineered['geohash'].apply(lambda x: pgh.decode(x)[0] if pd.notnull(x) else np.nan)
            df_engineered['longitude'] = df_engineered['geohash'].apply(lambda x: pgh.decode(x)[1] if pd.notnull(x) else np.nan)
        
        # 2. Temporal Features
        if 'timestamp' in df_engineered.columns:
            df_engineered['hour'] = df_engineered['timestamp'].str.split(':').str[0].astype(float)
            df_engineered['minute'] = df_engineered['timestamp'].str.split(':').str[1].astype(float)
            
            df_engineered['hour_sin'] = np.sin(2 * np.pi * df_engineered['hour'] / 24.0)
            df_engineered['hour_cos'] = np.cos(2 * np.pi * df_engineered['hour'] / 24.0)
            
            df_engineered['is_rush_hour'] = df_engineered['hour'].apply(
                lambda h: 1 if (7 <= h <= 10) or (16 <= h <= 19) else 0
            )
            df_engineered.drop(['timestamp'], axis=1, inplace=True)
            
        if 'day' in df_engineered.columns:
            df_engineered['day_of_week'] = df_engineered['day'].astype(int) % 7
            df_engineered['is_weekend'] = df_engineered['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
            
        # 3. Infrastructure & Environment Combinations
        if 'RoadType' in df_engineered.columns and 'NumberofLanes' in df_engineered.columns:
            df_engineered['road_capacity'] = df_engineered['RoadType'].astype(str) + "_lanes_" + df_engineered['NumberofLanes'].astype(str)
            
        if 'RoadType' in df_engineered.columns and 'Weather' in df_engineered.columns:
            df_engineered['weather_impact'] = df_engineered['Weather'].astype(str) + "_on_" + df_engineered['RoadType'].astype(str)
            
        if 'Temperature' in df_engineered.columns:
            df_engineered['temp_squared'] = df_engineered['Temperature'] ** 2
        
        # 4. Handle Missing Values
        num_cols = df_engineered.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = df_engineered.select_dtypes(include=['object']).columns
        
        for col in num_cols:
            if col != 'demand':
                df_engineered[col] = df_engineered[col].fillna(df_engineered[col].median())
                
        for col in cat_cols:
            df_engineered[col] = df_engineered[col].fillna('Unknown')
            
        return df_engineered

    def apply_clustering(self, X_train, X_test):
        """Helper method for the ensemble script."""
        print("Clustering Geographical Zones (n=15)...")
        coords_train = X_train[['latitude', 'longitude']].fillna(0)
        coords_test = X_test[['latitude', 'longitude']].fillna(0)
        
        X_train['traffic_zone'] = self.kmeans.fit_predict(coords_train).astype(str)
        X_test['traffic_zone'] = self.kmeans.predict(coords_test).astype(str)
        self.is_kmeans_fitted = True
        
        return X_train, X_test

    def train(self, X_train, y_train, X_val, y_val):
        """Satisfies the evaluator contract by training the local CatBoost model."""
        
        # 0. SAFETY RAIL: Drop the raw geohash before it crashes CatBoost
        X_train = X_train.drop(['geohash'], axis=1, errors='ignore')
        X_val = X_val.drop(['geohash'], axis=1, errors='ignore')

        # 1. Apply Spatial Clustering if not already done
        if 'traffic_zone' not in X_train.columns:
            X_train, X_val = self.apply_clustering(X_train, X_val)
            
        # 2. SAFETY RAIL: Bulletproof Categorical Detection
        cat_features = X_train.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        
        # print("Training Local CatBoost Engine...") # Turned off so it doesn't spam the tuner
        self.model.fit(
            X_train, y_train, 
            cat_features=cat_features, 
            eval_set=(X_val, y_val), 
            use_best_model=True
        )

    def predict(self, X_test):
        """Satisfies the evaluator contract for inference."""
        X_test_final = X_test.copy()
        
        # SAFETY RAIL: Drop geohash here too
        X_test_final = X_test_final.drop(['geohash'], axis=1, errors='ignore')
        
        # Apply Spatial Clustering to test set
        if 'traffic_zone' not in X_test_final.columns:
            coords_test = X_test_final[['latitude', 'longitude']].fillna(0)
            X_test_final['traffic_zone'] = self.kmeans.predict(coords_test).astype(str)
            
        # SAFETY RAIL: Missing data net for Hackathon robustness
        cat_cols = X_test_final.select_dtypes(include=['object', 'string', 'category']).columns
        for col in cat_cols:
            X_test_final[col] = X_test_final[col].fillna("Unknown").astype(str)
            
        preds = self.model.predict(X_test_final)
        return np.clip(preds, a_min=0, a_max=None)