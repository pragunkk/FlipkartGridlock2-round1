import pandas as pd
import numpy as np
import pygeohash as pgh
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

class TrafficModel:
    def __init__(self):
        # Increased iterations to 2500 to allow learning of complex infrastructure interactions
        self.model = CatBoostRegressor(
            iterations=2500,           
            learning_rate=0.010701616918559144,        
            depth=5,                   
            l2_leaf_reg=6.005961367727038,             
            bagging_temperature=0.5695929305875367,   
            random_strength=0.7699352773848305,       
            eval_metric='R2',
            random_seed=42,
            od_type='Iter',
            od_wait=100,               
            verbose=250  
        )
        self.categorical_features = None
        self.kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
        
        # NEW: Flag to ensure K-Means is only fitted once on the training set
        self.is_kmeans_fitted = False 

    def engineer_features(self, df):
        """
        Transforms raw data into machine-readable math.
        """
        df_engineered = df.copy()
        
        # 1. Spatial Features & Clustering Prep
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
            
        # 5. K-Means Clustering (MOVED HERE FROM PREDICT)
        if 'latitude' in df_engineered.columns and 'longitude' in df_engineered.columns:
            coords = df_engineered[['latitude', 'longitude']].fillna(df_engineered[['latitude', 'longitude']].median())
            
            if not self.is_kmeans_fitted:
                self.kmeans.fit(coords)
                self.is_kmeans_fitted = True
                
            # Convert to string so CatBoost treats it as a categorical feature
            df_engineered['traffic_zone'] = self.kmeans.predict(coords).astype(str)
            
        # 6. Drop raw strings to prevent CatBoost crashes
        df_engineered.drop(['geohash'], axis=1, inplace=True, errors='ignore')
        
        return df_engineered

    def train(self, X_train, y_train, X_val, y_val):
        """Finds categorical columns and trains the model."""
        
        # BULLETPROOF CATEGORICAL DETECTION (Do not overwrite this!)
        # Extract the actual column names instead of integer index positions
        cat_features = X_train.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        
        print(f"   -> Automatically detected categorical features: {cat_features}")
        
        self.model.fit(
            X_train, y_train, 
            cat_features=cat_features, 
            eval_set=(X_val, y_val), 
            use_best_model=True
        )

    def predict(self, X_test):
        """
        Generates predictions with a safety net for missing inference data.
        """
        X_test_safe = X_test.copy()
        
        # SAFETY NET: If the hidden test set drops categorical data (NaN), 
        # CatBoost will crash because it expects strings, not float NaNs.
        cat_cols = X_test_safe.select_dtypes(include=['object', 'string', 'category']).columns
        for col in cat_cols:
            X_test_safe[col] = X_test_safe[col].fillna("Unknown").astype(str)
            
        preds = self.model.predict(X_test_safe)
        return np.clip(preds, a_min=0, a_max=None)