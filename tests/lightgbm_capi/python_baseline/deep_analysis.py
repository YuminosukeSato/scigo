#!/usr/bin/env python3

import json
import numpy as np
import lightgbm as lgb

# Load test data
with open('../testdata/regression/test_data.json', 'r') as f:
    test_data = json.load(f)

# Load the model
model = lgb.Booster(model_file='../testdata/regression/model.txt')
sample = test_data['X_test'][0]

print("=== Deep Analysis of LightGBM Prediction ===")

# Get the training labels mean (should be init_score)
y_values = test_data['y_test']
init_score_estimate = np.mean(y_values)
print(f"Training labels mean: {init_score_estimate:.6f}")

# Compare with Python's base prediction
base_pred = model.predict([sample], num_iteration=0)[0]
print(f"Python base prediction (num_iteration=0): {base_pred:.6f}")
print(f"Difference: {base_pred - init_score_estimate:.6f}")

# Get tree contributions
try:
    contributions = model.predict([sample], pred_contrib=True)[0]
    print(f"\nTree contributions: {contributions}")
    print(f"Sum of contributions: {sum(contributions):.6f}")
    print(f"Expected total: {model.predict([sample])[0]:.6f}")
except Exception as e:
    print(f"Contributions not available: {e}")

# Manual calculation
print(f"\nManual verification:")
print(f"Base prediction: {base_pred:.6f}")

total = base_pred
for i in range(model.num_trees()):
    pred_i = model.predict([sample], num_iteration=i+1)[0]
    if i == 0:
        contribution = pred_i - base_pred
    else:
        contribution = pred_i - prev_pred
    
    total_check = base_pred + contribution if i == 0 else total_check + contribution
    print(f"Tree {i}: pred={pred_i:.6f}, contrib={contribution:.6f}, total={total_check:.6f}")
    prev_pred = pred_i

final_pred = model.predict([sample])[0]
print(f"\nFinal prediction: {final_pred:.6f}")

# Check if Tree 0 leaf value in file matches the raw contribution
print(f"\n=== Raw Tree Values vs Contributions ===")
print(f"Tree 0 raw leaf value (from Go): 11.009097")
print(f"Tree 0 contribution (Python): -63.247085")
print(f"Expected relationship: leaf_value + init_score_adjustment = contribution")
print(f"Calculation: 11.009097 - 74.256182 = {11.009097 - 74.256182:.6f}")

# Test: What if we consider the leaf value as the final prediction, not contribution?
print(f"\n=== Alternative interpretation ===")
print(f"What if Tree 0 leaf value (11.009097) is the FINAL prediction after applying init_score?")
print(f"Then contribution = 11.009097 - 74.256182 = {11.009097 - 74.256182:.6f}")
print(f"This matches Python's Tree 0 contribution: -63.247085 ✓")