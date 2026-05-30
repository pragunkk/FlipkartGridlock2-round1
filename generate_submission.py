import pandas as pd
import numpy as np
import warnings
from model_pipeline import TrafficModel

warnings.filterwarnings('ignore')

print("1. Loading datasets...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Ensure chronological order for training data just in case
train = train.sort_values(by=['day', 'timestamp'])

print("2. Initializing Tuned Pipeline...")
pipeline = TrafficModel()

print("3. Engineering Features for Train and Test...")
df_train_engineered = pipeline.engineer_features(train)
df_test_engineered = pipeline.engineer_features(test)

# Separate targets and features for training
y_train = df_train_engineered['demand']
X_train = df_train_engineered.drop(['Index', 'demand'], axis=1, errors='ignore')

# Prepare test set (Drop Index, ensure columns match)
X_test = df_test_engineered.drop(['Index', 'demand'], axis=1, errors='ignore')

print("4. Training Model on 100% of Training Data...")
# We pass X_train and y_train as the validation set as well just to satisfy 
# the pipeline's function signature, ensuring it trains on all available data!
pipeline.train(X_train, y_train, X_train, y_train)

print("5. Generating Final Predictions...")
final_predictions = pipeline.predict(X_test)

# 6. Save the Submission File
print("6. Saving new submission.csv...")
submission = pd.DataFrame({
    'Index': test['Index'],
    'demand': final_predictions
})

submission.to_csv('submission.csv', index=False)
print("SUCCESS: submission.csv has been updated with the latest architecture!")