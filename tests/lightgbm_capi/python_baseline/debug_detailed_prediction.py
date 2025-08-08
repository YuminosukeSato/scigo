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

# Get detailed tree-by-tree contributions
print(f"\nDetailed tree-by-tree analysis:")
cumsum = 0.0

# Get initial score (base_score)
# LightGBM's init_score is typically 0, but let's see what we get
pred_base = model.predict([sample], num_iteration=0)
if len(pred_base) > 0:
    print(f"Base prediction (0 trees): {pred_base[0]}")
    cumsum = pred_base[0]
else:
    print("No base prediction available")

for i in range(model.num_trees()):
    pred_i = model.predict([sample], num_iteration=i+1)[0]
    if i == 0:
        tree_contrib = pred_i - cumsum
    else:
        tree_contrib = pred_i - prev_pred
    cumsum += tree_contrib if i == 0 else 0
    print(f"  Tree {i}: contrib={tree_contrib:.6f}, total={pred_i:.6f}")
    prev_pred = pred_i

print(f"\nFinal prediction: {model.predict([sample])[0]}")

# Also get leaf indices to compare with Go implementation
pred_leaf = model.predict([sample], pred_leaf=True)
print(f"Leaf indices: {pred_leaf[0]}")

# Try to understand the init_score situation
print(f"\nModel parameters:")
params = model.params
if 'init_score' in params:
    print(f"  init_score in params: {params['init_score']}")
else:
    print("  No init_score in params")

# Check if we can get the actual base value
try:
    dump_model = model.dump_model()
    if 'init_score' in dump_model:
        print(f"  init_score in dump: {dump_model['init_score']}")
    else:
        print("  No init_score in dump")
except:
    print("  Could not dump model")