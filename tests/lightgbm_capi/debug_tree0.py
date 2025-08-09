#!/usr/bin/env python3

import json
import numpy as np
import lightgbm as lgb

# Load the model and analyze Tree 0 specifically
model = lgb.Booster(model_file='testdata/regression/model.txt')
dump = model.dump_model()

print("=== Tree 0 Analysis ===")
tree0 = dump['tree_info'][0]
print(f"Tree 0 keys: {list(tree0.keys())}")
print(f"Tree 0 shrinkage: {tree0['shrinkage']}")

# Load test data
with open('testdata/regression/test_data.json', 'r') as f:
    test_data = json.load(f)

sample = test_data['X_test'][0]

# Check what LightGBM does with just tree 0
pred_0trees = model.predict([sample], num_iteration=0)[0]  # Base prediction
pred_1tree = model.predict([sample], num_iteration=1)[0]   # Base + Tree 0

print(f"\nPrediction analysis:")
print(f"Base (0 trees): {pred_0trees}")  
print(f"After tree 0 (1 tree): {pred_1tree}")
print(f"Tree 0 contribution: {pred_1tree - pred_0trees}")

# Try to understand the raw tree 0 leaf value
pred_leaf = model.predict([sample], pred_leaf=True)
tree0_leaf_idx = pred_leaf[0][0]
print(f"Tree 0 reaches leaf index: {tree0_leaf_idx}")

# Try to manually walk tree 0 using the dumped structure
def walk_tree(node, features):
    print(f"Node - feature: {node.get('split_feature', 'LEAF')}, threshold: {node.get('threshold', 'N/A')}")
    
    if 'left_child' not in node:  # Leaf node
        print(f"LEAF VALUE: {node.get('leaf_value', 'N/A')}")
        return node.get('leaf_value', 0)
    
    feature_idx = node['split_feature']
    threshold = node['threshold']
    feature_val = features[feature_idx]
    
    print(f"  Comparing {feature_val} <= {threshold}")
    
    if feature_val <= threshold:
        print("  -> Going LEFT")
        return walk_tree(node['left_child'], features)
    else:
        print("  -> Going RIGHT") 
        return walk_tree(node['right_child'], features)

print(f"\nManual tree traversal:")
raw_leaf_value = walk_tree(tree0['tree_structure'], sample)
print(f"Raw leaf value: {raw_leaf_value}")
print(f"Shrinkage applied: {raw_leaf_value * tree0['shrinkage']}")

# Check if there's any transformation applied
print(f"Expected contribution: {pred_1tree - pred_0trees}")