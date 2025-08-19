#!/usr/bin/env python3

import lightgbm as lgb
import numpy as np

# Load model
model = lgb.Booster(model_file='testdata/compatibility/regression_model.txt')

# Test features - first row from regression_X_test.csv
X = np.array([[
    1.5904035685736386, -0.3939866812431204, 0.04092474872892285,
    -0.9984408528767182, 1.9913704184169045, 0.43494103836789866,
    1.6232566905280539, -0.5691481986725339, -0.7971135662249808,
    0.3929135105482163
]])

# Get raw predictions (without init_score)
pred_raw = model.predict(X, raw_score=True)
print(f"Raw prediction (with init_score): {pred_raw[0]}")

# Get prediction
pred_normal = model.predict(X)
print(f"Normal prediction: {pred_normal[0]}")

# Try to get model info
try:
    # Attempt different methods to get init_score
    # Option 1: From model string
    model_str = model.model_to_string()
    lines = model_str.split('\n')
    for line in lines[:50]:  # Check first 50 lines
        if 'init_score' in line.lower() or 'average' in line.lower() or 'init' in line.lower():
            print(f"Found potential init line: {line}")
except Exception as e:
    print(f"Error getting model string: {e}")

# Check if first tree contains init_score
print(f"\nLet's check tree predictions:")
for i in [1, 2, 3, 5, 10, 100]:
    pred = model.predict(X, num_iteration=i, raw_score=True)
    print(f"Raw score with {i} trees: {pred[0]}")
    
# Calculate difference
print(f"\nDifference between 1 tree and 2 trees: {model.predict(X, num_iteration=2)[0] - model.predict(X, num_iteration=1)[0]}")