#!/usr/bin/env python3
"""
Create a very simple LightGBM model for debugging purposes.
"""

import numpy as np
import lightgbm as lgb
import json

# Create very simple data
np.random.seed(42)
X_train = np.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1.0]])
y_train = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

X_test = np.array([[0.25], [0.75]])
y_test = np.array([0.25, 0.75])

# Train a very simple model
params = {
    'objective': 'regression',
    'num_leaves': 2,  # Minimal tree
    'learning_rate': 1.0,  # No shrinkage for first tree
    'min_data_in_leaf': 1,
    'verbose': 1,
    'seed': 42,
    'force_col_wise': True,
}

train_data = lgb.Dataset(X_train, label=y_train)
model = lgb.train(params, train_data, num_boost_round=1)  # Only 1 tree

# Save model
model.save_model('testdata/simple_model.txt')

# Make predictions
predictions = model.predict(X_test)
print(f"Test data: {X_test.flatten()}")
print(f"Predictions: {predictions}")

# Save test data and predictions
np.savetxt('testdata/simple_X_test.csv', X_test, delimiter=',', fmt='%.10f')
np.savetxt('testdata/simple_predictions.csv', predictions, delimiter=',', fmt='%.10f')

# Dump model structure for debugging
model_json = model.dump_model()
with open('testdata/simple_model.json', 'w') as f:
    json.dump(model_json, f, indent=2)

print("\nModel structure:")
print(f"Number of trees: {model.num_trees()}")
print(f"Number of features: {model.num_feature()}")

# Print tree structure
tree_info = model_json['tree_info'][0]['tree_structure']
print("\nTree structure:")
print(json.dumps(tree_info, indent=2))