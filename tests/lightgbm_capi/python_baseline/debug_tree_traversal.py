#!/usr/bin/env python3

import json
import numpy as np
import lightgbm as lgb

# Load test data
with open('../testdata/regression/test_data.json', 'r') as f:
    test_data = json.load(f)

# Load the model
model = lgb.Booster(model_file='../testdata/regression/model.txt')

# Get first sample
sample = test_data['X_test'][0]
print(f"Sample features: {sample}")
print(f"Expected prediction: {test_data['predictions'][0]}")

# Get model parameters
print(f"\nModel info:")
print(f"  Num trees: {model.num_trees()}")

# Debug tree by tree predictions (using predict with num_iteration parameter)
cumsum = 0.0
print(f"\nTree-by-tree predictions:")

# Tree 0
pred0 = model.predict([sample], num_iteration=1)[0]
print(f"  After tree 0: {pred0}")

# Tree 1 
pred1 = model.predict([sample], num_iteration=2)[0]
tree1_contrib = pred1 - pred0
print(f"  After tree 1: {pred1} (tree 1 contrib: {tree1_contrib})")

# Tree 2
pred2 = model.predict([sample], num_iteration=3)[0] 
tree2_contrib = pred2 - pred1
print(f"  After tree 2: {pred2} (tree 2 contrib: {tree2_contrib})")

# Use leaf prediction to get actual leaf indices (if available)
try:
    # This might show which leaf each tree reaches
    pred_leaf = model.predict([sample], pred_leaf=True)
    print(f"\nLeaf indices per tree: {pred_leaf[0]}")
except:
    print("Leaf prediction not available")

print(f"\nFinal prediction: {model.predict([sample])[0]}")