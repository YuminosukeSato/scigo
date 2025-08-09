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

print("=== Final Analysis ===")
print("Focus: Understanding what each tree actually produces")

# Get step by step predictions
predictions_by_iteration = []
for i in range(model.num_trees() + 1):  # Include num_iteration=0
    pred = model.predict([sample], num_iteration=i)[0]
    predictions_by_iteration.append(pred)

print(f"\nStep-by-step predictions:")
for i, pred in enumerate(predictions_by_iteration):
    print(f"  num_iteration={i}: {pred:.6f}")

# Calculate what each tree contributes
print(f"\nTree contributions (calculated):")
for i in range(1, len(predictions_by_iteration)):
    contrib = predictions_by_iteration[i] - predictions_by_iteration[i-1]
    print(f"  Tree {i-1}: {contrib:.6f}")

# Get raw contributions if available
try:
    raw_contribs = model.predict([sample], pred_contrib=True)[0]
    print(f"\nRaw contributions from predict(pred_contrib=True):")
    for i, contrib in enumerate(raw_contribs):
        if i < 10:  # Only show first 10 trees
            print(f"  Tree {i}: {contrib:.6f}")
    print(f"  Sum: {sum(raw_contribs):.6f}")
except Exception as e:
    print(f"Raw contributions not available: {e}")

# The key insight: What does num_iteration=0 represent?
base_pred = predictions_by_iteration[0]  # num_iteration=0
final_pred = predictions_by_iteration[-1]  # All trees

print(f"\n=== Key Insights ===")
print(f"num_iteration=0 (base): {base_pred:.6f}")
print(f"All trees (final): {final_pred:.6f}")
print(f"Total tree contribution: {final_pred - base_pred:.6f}")

# This suggests the model structure is:
# final_prediction = base_prediction + tree_contributions
# where base_prediction comes from somewhere (training avg, init_score, etc)
# and tree_contributions are learned by the actual trees

# Let's verify: If we start from 0 and add all tree contributions, do we get final - base?
tree_contribs = []
for i in range(1, len(predictions_by_iteration)):
    contrib = predictions_by_iteration[i] - predictions_by_iteration[i-1]
    tree_contribs.append(contrib)

total_tree_contrib = sum(tree_contribs)
print(f"\nVerification:")
print(f"Sum of tree contributions: {total_tree_contrib:.6f}")
print(f"Final - Base: {final_pred - base_pred:.6f}")
print(f"Match: {abs(total_tree_contrib - (final_pred - base_pred)) < 1e-10}")

print(f"\n=== For Go Implementation ===")
print(f"The leaf values in the model file represent CONTRIBUTIONS, not absolute values")
print(f"Go should:")
print(f"1. Set init_score = {base_pred:.6f} (base prediction)")
print(f"2. Add each tree's leaf value directly as contribution")
print(f"3. final_prediction = init_score + sum(tree_leaf_values)")

# Let's see what the actual leaf value is for Tree 0
print(f"\nTree 0 analysis:")
print(f"  Go gets leaf value: 11.009097")
print(f"  Python tree 0 contribution: {tree_contribs[0]:.6f}")
print(f"  These should match if leaf values are contributions!")
print(f"  Match: {abs(11.009097 - tree_contribs[0]) < 1e-6}")