#!/usr/bin/env python3

import json
import numpy as np
import lightgbm as lgb

# Load the model
model = lgb.Booster(model_file='../testdata/regression/model.txt')

print("=== Model Analysis ===")
print(f"Num trees: {model.num_trees()}")

# Try to get the raw model dump
try:
    dump = model.dump_model()
    print("\nModel dump keys:", list(dump.keys()))
    
    if 'tree_info' in dump:
        print(f"Number of trees in dump: {len(dump['tree_info'])}")
        
        # Look at first tree structure
        tree0 = dump['tree_info'][0]
        print(f"\nTree 0 keys: {list(tree0.keys())}")
        
        if 'tree_structure' in tree0:
            root = tree0['tree_structure']
            print(f"Root node keys: {list(root.keys())}")
            if 'internal_value' in root:
                print(f"Root internal_value: {root['internal_value']}")
                
        # Check for any init_score or similar
        for key in tree0.keys():
            if 'init' in key.lower() or 'score' in key.lower() or 'value' in key.lower():
                print(f"Tree 0 {key}: {tree0[key]}")
                
    # Check overall model for any base/init values
    for key in dump.keys():
        if 'init' in key.lower() or 'base' in key.lower() or 'score' in key.lower():
            print(f"Model {key}: {dump[key]}")
            
except Exception as e:
    print(f"Error dumping model: {e}")

# Test prediction with manual tree traversal simulation
sample = [0.3319184477078873, 0.14729800038908406, 1.6814924574180155, -0.8838688951407293, -1.027923765010739, 0.6149446390313781, 0.8117055716132655, -0.6804940477640479, 0.509683077242329, 2.935406206388468]

# Try to understand what happens with different num_iteration values
print(f"\n=== Prediction Analysis ===")
for i in range(3):
    pred = model.predict([sample], num_iteration=i)
    print(f"num_iteration={i}: {pred[0] if len(pred) > 0 else 'N/A'}")

# Get the leaf predictions to see raw tree outputs
print(f"\n=== Leaf Analysis ===")
pred_leaf = model.predict([sample], pred_leaf=True)
print(f"Leaf indices: {pred_leaf[0]}")

# Try to get contribution per tree
try:
    pred_contrib = model.predict([sample], pred_contrib=True)
    print(f"Tree contributions shape: {np.array(pred_contrib).shape}")
    print(f"Tree contributions: {pred_contrib[0]}")
except:
    print("pred_contrib not available")