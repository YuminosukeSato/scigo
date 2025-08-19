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

# Get prediction
pred = model.predict(X)
print(f"Final prediction: {pred[0]}")

# Test prediction with different num_iteration
for num_iter in [1, 2, 3, 5, 10, 100]:
    pred = model.predict(X, num_iteration=num_iter)
    print(f"Prediction with {num_iter} trees: {pred[0]}")

# Get tree info
model_json = model.dump_model()
print(f"\nFirst tree shrinkage: {model_json['tree_info'][0]['shrinkage']}")
print(f"Second tree shrinkage: {model_json['tree_info'][1]['shrinkage']}")