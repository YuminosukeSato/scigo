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

# Get prediction for just the first tree
pred_1tree = model.predict(X, num_iteration=1)
print(f"Prediction with 1 tree: {pred_1tree[0]}")

# Get leaf indices
leaf_indices = model.predict(X, pred_leaf=True)
print(f"Leaf indices shape: {leaf_indices.shape}")
print(f"First tree leaf index: {leaf_indices[0, 0]}")

# Get model dump
model_json = model.dump_model()
first_tree_info = model_json['tree_info'][0]
print(f"\nFirst tree info:")
print(f"  Num leaves: {first_tree_info['num_leaves']}")
print(f"  Shrinkage: {first_tree_info['shrinkage']}")

# Get init score
print(f"\nInit score (from Python): {model_json.get('init_score', 'Not found')}")

# Manually trace through first tree
def trace_tree(tree_structure, features):
    node = tree_structure
    path = []
    
    while 'left_child' in node:
        feature_idx = node['split_feature']
        threshold = node['threshold']
        feature_val = features[feature_idx]
        
        path.append(f"Node: feature[{feature_idx}]={feature_val:.4f} {'<=' if feature_val <= threshold else '>'} {threshold:.4f}")
        
        if feature_val <= threshold:
            node = node['left_child']
        else:
            node = node['right_child']
    
    # Reached a leaf
    leaf_value = node['leaf_value']
    leaf_index = node['leaf_index']
    path.append(f"Leaf {leaf_index}: value={leaf_value}")
    return leaf_index, leaf_value, path

tree_structure = model_json['tree_structure'][0]['tree_structure']
leaf_idx, leaf_val, path = trace_tree(tree_structure, X[0])

print(f"\nManual tree traversal:")
for p in path:
    print(f"  {p}")
    
print(f"\nLeaf value from manual traversal: {leaf_val}")
print(f"Expected output for 1 tree: {pred_1tree[0]}")
print(f"Difference: {pred_1tree[0] - leaf_val}")