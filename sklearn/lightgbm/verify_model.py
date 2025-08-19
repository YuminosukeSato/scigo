#!/usr/bin/env python3

import lightgbm as lgb
import numpy as np

# Load model
model = lgb.Booster(model_file='testdata/compatibility/regression_model.txt')

# Test features
X = np.array([[0.5, -0.5, 1.0, -1.0, 0.0, 0.2, -0.2, 0.8, -0.8, 0.3]])

# Get prediction
pred = model.predict(X)
print(f"Final prediction: {pred[0]}")

# Get predictions tree by tree
print(f"\nNumber of trees: {model.num_trees()}")
print(f"Best iteration: {model.best_iteration}")

# Get model dump
model_json = model.dump_model()
print(f"Number of tree_info entries: {len(model_json['tree_info'])}")

# Check first few trees
for i in range(min(3, len(model_json['tree_info']))):
    tree = model_json['tree_info'][i]
    print(f"\nTree {i}:")
    print(f"  Num leaves: {tree['num_leaves']}")
    print(f"  Shrinkage: {tree['shrinkage']}")

# Test prediction with different num_iteration
for num_iter in [1, 2, 3, 5, 10, 100]:
    pred = model.predict(X, num_iteration=num_iter)
    print(f"Prediction with {num_iter} trees: {pred[0]}")