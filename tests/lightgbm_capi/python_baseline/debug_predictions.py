#!/usr/bin/env python3
"""Debug LightGBM predictions to understand internal values."""

import json
import numpy as np
import lightgbm as lgb

# Load model
model = lgb.Booster(model_file="../testdata/regression/model.txt")

# Load test data
with open("../testdata/regression/test_data.json", 'r') as f:
    test_data = json.load(f)

X_test = np.array(test_data['X_test'])
y_test = np.array(test_data['y_test'])
expected_preds = np.array(test_data['predictions'])

# Get first sample
sample = X_test[0]
print("First sample features:")
for i, v in enumerate(sample):
    print(f"  Feature {i}: {v:.4f}")
print()

# Make prediction
pred = model.predict([sample])[0]
print(f"Python prediction: {pred:.6f}")
print(f"Expected from JSON: {expected_preds[0]:.6f}")
print()

# Get model details
print("Model Info:")
print(f"  Objective: {model.params.get('objective', 'unknown')}")
print(f"  Num trees: {model.num_trees()}")
print(f"  Num features: {model.num_feature()}")
print()

# Try to get init score
dump = model.dump_model()
print(f"Tree 0 root internal value: {dump['tree_info'][0].get('tree_structure', {}).get('internal_value', 'N/A')}")

# Get raw predictions (leaf values) for each tree
print("\nTree-by-tree predictions:")
cumsum = 0.0
for i in range(model.num_trees()):
    # Get prediction from single tree
    tree_pred = model.predict([sample], start_iteration=i, num_iteration=1)[0]
    if i == 0:
        raw_pred = tree_pred
        cumsum = tree_pred
    else:
        raw_pred = tree_pred - cumsum
        cumsum = tree_pred
    
    # Get tree info
    tree_info = dump['tree_info'][i]
    shrinkage = tree_info.get('shrinkage', 1.0)
    
    print(f"  Tree {i}: raw={raw_pred:.6f}, shrinkage={shrinkage:.2f}, cum={cumsum:.6f}")
    
    if i >= 9:  # Only show first 10 trees
        break

print(f"\nFinal prediction: {pred:.6f}")

# Analyze tree structure for first tree
print("\n\nFirst tree structure analysis:")
tree_struct = dump['tree_info'][0]['tree_structure']

def print_tree_node(node, depth=0):
    indent = "  " * depth
    if 'leaf_value' in node:
        print(f"{indent}Leaf: value={node['leaf_value']:.6f}")
    else:
        print(f"{indent}Node: split_feature={node['split_feature']}, threshold={node['threshold']:.6f}")
        print_tree_node(node['left_child'], depth + 1)
        print_tree_node(node['right_child'], depth + 1)

print_tree_node(tree_struct, 0)