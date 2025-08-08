#!/usr/bin/env python3
"""
Generate comprehensive test data for verifying Go implementation compatibility with LightGBM C API.
This script generates deterministic training results and intermediate computation details.
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Ensure deterministic results
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Output directory
OUTPUT_DIR = "../testdata/compatibility"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_tree_structure(booster):
    """Extract detailed tree structure from LightGBM model."""
    model_dump = booster.dump_model()
    trees = []
    
    for tree_info in model_dump['tree_info']:
        tree = {
            'tree_index': tree_info['tree_index'],
            'num_leaves': tree_info['num_leaves'],
            'num_nodes': len(tree_info['tree_structure']) if 'tree_structure' in tree_info else 0,
            'shrinkage': tree_info.get('shrinkage', 1.0),
        }
        
        # Extract node information recursively
        if 'tree_structure' in tree_info:
            nodes = []
            def traverse_node(node, node_id=0):
                node_info = {
                    'node_id': node_id,
                    'is_leaf': 'leaf_value' in node,
                }
                
                if 'leaf_value' in node:
                    node_info['leaf_value'] = node['leaf_value']
                    node_info['leaf_count'] = node.get('leaf_count', 0)
                else:
                    node_info['split_feature'] = node['split_feature']
                    node_info['threshold'] = node['threshold']
                    node_info['decision_type'] = node['decision_type']
                    node_info['default_left'] = node.get('default_left', True)
                    node_info['missing_type'] = node.get('missing_type', 'None')
                    node_info['internal_value'] = node.get('internal_value', 0)
                    node_info['internal_count'] = node.get('internal_count', 0)
                    node_info['split_gain'] = node.get('split_gain', 0)
                    
                    # Recursively process children
                    if 'left_child' in node:
                        left_id = len(nodes) + 1
                        node_info['left_child'] = left_id
                        nodes.append(node_info)
                        traverse_node(node['left_child'], left_id)
                    
                    if 'right_child' in node:
                        right_id = len(nodes) + 1
                        node_info['right_child'] = right_id
                        if 'left_child' not in node:
                            nodes.append(node_info)
                        traverse_node(node['right_child'], right_id)
                    
                    if 'left_child' not in node and 'right_child' not in node:
                        nodes.append(node_info)
                
                if not any(n['node_id'] == node_id for n in nodes):
                    nodes.append(node_info)
                    
            traverse_node(tree_info['tree_structure'])
            tree['nodes'] = nodes
            
        trees.append(tree)
    
    return trees

def save_histogram_data(booster, X_train, y_train, output_path):
    """Extract and save histogram construction details."""
    # This would require accessing LightGBM internals
    # For now, we'll save the data that would be used to construct histograms
    histogram_data = {
        'num_features': X_train.shape[1],
        'num_samples': X_train.shape[0],
        'feature_stats': []
    }
    
    for i in range(X_train.shape[1]):
        feature_vals = X_train[:, i]
        histogram_data['feature_stats'].append({
            'feature_index': i,
            'min': float(np.min(feature_vals)),
            'max': float(np.max(feature_vals)),
            'mean': float(np.mean(feature_vals)),
            'std': float(np.std(feature_vals)),
            'unique_values': int(len(np.unique(feature_vals)))
        })
    
    with open(output_path, 'w') as f:
        json.dump(histogram_data, f, indent=2)

def generate_regression_test():
    """Generate regression test data with detailed information."""
    print("Generating regression test data...")
    
    # Generate deterministic data
    X, y = make_regression(
        n_samples=500,
        n_features=10,
        n_informative=8,
        noise=0.1,
        random_state=RANDOM_STATE
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Save raw data
    np.save(f"{OUTPUT_DIR}/regression_X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/regression_X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/regression_y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/regression_y_test.npy", y_test)
    
    # Create LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Strict deterministic parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 5,
        'learning_rate': 0.1,
        'feature_fraction': 1.0,  # No feature sampling
        'bagging_fraction': 1.0,  # No data sampling
        'bagging_freq': 0,
        'min_data_in_leaf': 20,
        'min_sum_hessian_in_leaf': 0.001,
        'lambda_l2': 0.1,
        'lambda_l1': 0.0,
        'min_gain_to_split': 0.01,
        'verbosity': 2,
        'seed': RANDOM_STATE,
        'deterministic': True,
        'force_row_wise': True,
        'force_col_wise': False,
        'num_threads': 1,
        'max_bin': 255,
        'min_data_in_bin': 3,
        'bin_construct_sample_cnt': 200000,
        'histogram_pool_size': -1,
    }
    
    # Train model
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=10,
        valid_sets=[train_data],
        callbacks=[lgb.log_evaluation(1)]
    )
    
    # Save model
    booster.save_model(f"{OUTPUT_DIR}/regression_model.txt")
    
    # Extract and save tree structure
    tree_structure = extract_tree_structure(booster)
    with open(f"{OUTPUT_DIR}/regression_tree_structure.json", 'w') as f:
        json.dump(tree_structure, f, indent=2)
    
    # Save predictions
    train_pred = booster.predict(X_train, num_iteration=booster.best_iteration)
    test_pred = booster.predict(X_test, num_iteration=booster.best_iteration)
    
    # Save histogram data
    save_histogram_data(booster, X_train, y_train, 
                       f"{OUTPUT_DIR}/regression_histogram_data.json")
    
    # Compile all test data
    test_data = {
        'params': params,
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'num_trees': booster.num_trees(),
        'feature_importance': booster.feature_importance().tolist(),
        'train_predictions': train_pred.tolist(),
        'test_predictions': test_pred.tolist(),
        'init_score': float(np.mean(y_train)),  # For regression, init score is mean
        'objective': 'regression',
    }
    
    with open(f"{OUTPUT_DIR}/regression_test_data.json", 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Regression test data saved to {OUTPUT_DIR}")
    return booster, X_test, test_pred

def generate_binary_test():
    """Generate binary classification test data."""
    print("Generating binary classification test data...")
    
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=RANDOM_STATE
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Save raw data
    np.save(f"{OUTPUT_DIR}/binary_X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/binary_X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/binary_y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/binary_y_test.npy", y_test)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 5,
        'learning_rate': 0.1,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'bagging_freq': 0,
        'min_data_in_leaf': 20,
        'lambda_l2': 0.1,
        'lambda_l1': 0.0,
        'min_gain_to_split': 0.01,
        'verbosity': 2,
        'seed': RANDOM_STATE,
        'deterministic': True,
        'force_row_wise': True,
        'num_threads': 1,
        'max_bin': 255,
    }
    
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=10,
        valid_sets=[train_data],
        callbacks=[lgb.log_evaluation(1)]
    )
    
    booster.save_model(f"{OUTPUT_DIR}/binary_model.txt")
    
    # Extract tree structure
    tree_structure = extract_tree_structure(booster)
    with open(f"{OUTPUT_DIR}/binary_tree_structure.json", 'w') as f:
        json.dump(tree_structure, f, indent=2)
    
    # Predictions
    train_pred_raw = booster.predict(X_train, num_iteration=booster.best_iteration)
    test_pred_raw = booster.predict(X_test, num_iteration=booster.best_iteration)
    
    test_data = {
        'params': params,
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'num_trees': booster.num_trees(),
        'feature_importance': booster.feature_importance().tolist(),
        'train_predictions': train_pred_raw.tolist(),
        'test_predictions': test_pred_raw.tolist(),
        'init_score': float(np.log(np.mean(y_train) / (1 - np.mean(y_train)))),  # logit for binary
        'objective': 'binary',
    }
    
    with open(f"{OUTPUT_DIR}/binary_test_data.json", 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Binary test data saved to {OUTPUT_DIR}")
    return booster, X_test, test_pred_raw

def generate_multiclass_test():
    """Generate multiclass classification test data."""
    print("Generating multiclass classification test data...")
    
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=2,
        random_state=RANDOM_STATE
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Save raw data
    np.save(f"{OUTPUT_DIR}/multiclass_X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/multiclass_X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/multiclass_y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/multiclass_y_test.npy", y_test)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 3,
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 5,
        'learning_rate': 0.1,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'bagging_freq': 0,
        'min_data_in_leaf': 20,
        'lambda_l2': 0.1,
        'lambda_l1': 0.0,
        'min_gain_to_split': 0.01,
        'verbosity': 2,
        'seed': RANDOM_STATE,
        'deterministic': True,
        'force_row_wise': True,
        'num_threads': 1,
        'max_bin': 255,
    }
    
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=10,
        valid_sets=[train_data],
        callbacks=[lgb.log_evaluation(1)]
    )
    
    booster.save_model(f"{OUTPUT_DIR}/multiclass_model.txt")
    
    # Extract tree structure
    tree_structure = extract_tree_structure(booster)
    with open(f"{OUTPUT_DIR}/multiclass_tree_structure.json", 'w') as f:
        json.dump(tree_structure, f, indent=2)
    
    # Predictions (returns probabilities for each class)
    train_pred_raw = booster.predict(X_train, num_iteration=booster.best_iteration)
    test_pred_raw = booster.predict(X_test, num_iteration=booster.best_iteration)
    
    test_data = {
        'params': params,
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'num_trees': booster.num_trees(),
        'num_class': 3,
        'feature_importance': booster.feature_importance().tolist(),
        'train_predictions': train_pred_raw.tolist(),
        'test_predictions': test_pred_raw.tolist(),
        'objective': 'multiclass',
    }
    
    with open(f"{OUTPUT_DIR}/multiclass_test_data.json", 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Multiclass test data saved to {OUTPUT_DIR}")
    return booster, X_test, test_pred_raw

def main():
    """Generate all test data."""
    print("="*60)
    print("Generating comprehensive C API compatibility test data")
    print("="*60)
    
    # Generate test data for each objective
    reg_booster, reg_X_test, reg_pred = generate_regression_test()
    bin_booster, bin_X_test, bin_pred = generate_binary_test()
    mc_booster, mc_X_test, mc_pred = generate_multiclass_test()
    
    # Summary
    print("\n" + "="*60)
    print("Test data generation complete!")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - *_model.txt: LightGBM model files")
    print("  - *_tree_structure.json: Detailed tree structure")
    print("  - *_test_data.json: Complete test data with predictions")
    print("  - *.npy: Raw numpy arrays of training/test data")
    print("\nModels summary:")
    print(f"  - Regression: {reg_booster.num_trees()} trees")
    print(f"  - Binary: {bin_booster.num_trees()} trees")
    print(f"  - Multiclass: {mc_booster.num_trees()} trees")

if __name__ == "__main__":
    main()