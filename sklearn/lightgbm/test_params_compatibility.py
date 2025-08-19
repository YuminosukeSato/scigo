#!/usr/bin/env python3
"""
Test parameter compatibility between Python LightGBM and Go implementation.
"""

import json
import numpy as np
import lightgbm as lgb
from pathlib import Path


def test_feature_fraction():
    """Test feature fraction parameter effect."""
    print("Testing feature_fraction parameter...")
    
    # Create dataset
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Test different feature fractions
    for ff in [0.3, 0.5, 0.7, 1.0]:
        params = {
            'objective': 'binary',
            'num_leaves': 5,
            'learning_rate': 0.1,
            'num_iterations': 3,
            'feature_fraction': ff,
            'bagging_fraction': 1.0,
            'lambda_l2': 1.0,
            'lambda_l1': 0.0,
            'min_data_in_leaf': 5,
            'seed': 42,
            'verbose': -1,
            'force_col_wise': True,
            'deterministic': True
        }
        
        train_data = lgb.Dataset(X, label=y)
        model = lgb.train(params, train_data)
        
        # Check that different features are used
        feature_importance = model.feature_importance(importance_type='split')
        features_used = np.sum(feature_importance > 0)
        
        print(f"  feature_fraction={ff}: {features_used}/10 features used")
        
        # Save model for Go comparison
        model.save_model(f'test_models/feature_fraction_{ff}.json')


def test_bagging():
    """Test bagging parameters."""
    print("Testing bagging parameters...")
    
    # Create dataset
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)
    
    # Test different bagging configurations
    configs = [
        {'bagging_fraction': 0.5, 'bagging_freq': 1},
        {'bagging_fraction': 0.7, 'bagging_freq': 2},
        {'bagging_fraction': 1.0, 'bagging_freq': 0},
    ]
    
    for config in configs:
        params = {
            'objective': 'binary',
            'num_leaves': 5,
            'learning_rate': 0.1,
            'num_iterations': 5,
            'feature_fraction': 1.0,
            'bagging_fraction': config['bagging_fraction'],
            'bagging_freq': config['bagging_freq'],
            'lambda_l2': 1.0,
            'min_data_in_leaf': 5,
            'seed': 42,
            'verbose': -1,
            'force_col_wise': True,
            'deterministic': True
        }
        
        train_data = lgb.Dataset(X, label=y)
        model = lgb.train(params, train_data)
        
        print(f"  bagging_fraction={config['bagging_fraction']}, "
              f"bagging_freq={config['bagging_freq']}: "
              f"{model.num_trees()} trees built")
        
        # Save model for comparison
        model.save_model(f"test_models/bagging_{config['bagging_fraction']}_{config['bagging_freq']}.json")


def test_regularization():
    """Test L1/L2 regularization parameters."""
    print("Testing L1/L2 regularization...")
    
    # Create dataset with potential overfitting
    np.random.seed(42)
    X = np.random.randn(50, 20)  # Many features, few samples
    y = (X[:, 0] + X[:, 1] + np.random.randn(50) * 0.1 > 0).astype(int)
    
    # Test different regularization settings
    configs = [
        {'lambda_l1': 0.0, 'lambda_l2': 0.0},
        {'lambda_l1': 0.0, 'lambda_l2': 10.0},
        {'lambda_l1': 5.0, 'lambda_l2': 0.0},
        {'lambda_l1': 5.0, 'lambda_l2': 5.0},
    ]
    
    for config in configs:
        params = {
            'objective': 'binary',
            'num_leaves': 10,
            'learning_rate': 0.1,
            'num_iterations': 5,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'lambda_l1': config['lambda_l1'],
            'lambda_l2': config['lambda_l2'],
            'min_data_in_leaf': 3,
            'seed': 42,
            'verbose': -1,
            'force_col_wise': True,
            'deterministic': True
        }
        
        train_data = lgb.Dataset(X, label=y)
        model = lgb.train(params, train_data)
        
        # Get leaf values to check regularization effect
        dump = model.dump_model()
        leaf_values = []
        for tree_info in dump['tree_info']:
            tree_struct = tree_info['tree_structure']
            leaves = extract_leaf_values(tree_struct)
            leaf_values.extend(leaves)
        
        avg_leaf_value = np.mean(np.abs(leaf_values)) if leaf_values else 0
        
        print(f"  L1={config['lambda_l1']}, L2={config['lambda_l2']}: "
              f"avg |leaf value|={avg_leaf_value:.4f}")
        
        # Save model
        model.save_model(f"test_models/regularization_l1_{config['lambda_l1']}_l2_{config['lambda_l2']}.json")


def extract_leaf_values(node):
    """Extract all leaf values from a tree structure."""
    if 'leaf_value' in node:
        return [node['leaf_value']]
    
    leaves = []
    if 'left_child' in node:
        leaves.extend(extract_leaf_values(node['left_child']))
    if 'right_child' in node:
        leaves.extend(extract_leaf_values(node['right_child']))
    
    return leaves


def test_min_data_in_leaf():
    """Test min_data_in_leaf parameter."""
    print("Testing min_data_in_leaf parameter...")
    
    # Create dataset
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)
    
    for min_data in [1, 10, 20, 30]:
        params = {
            'objective': 'binary',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'num_iterations': 3,
            'min_data_in_leaf': min_data,
            'lambda_l2': 1.0,
            'seed': 42,
            'verbose': -1,
            'force_col_wise': True,
            'deterministic': True
        }
        
        train_data = lgb.Dataset(X, label=y)
        model = lgb.train(params, train_data)
        
        # Count actual leaves
        dump = model.dump_model()
        total_leaves = 0
        for tree_info in dump['tree_info']:
            total_leaves += tree_info['num_leaves']
        
        print(f"  min_data_in_leaf={min_data}: {total_leaves} total leaves")
        
        # Save model
        model.save_model(f'test_models/min_data_in_leaf_{min_data}.json')


def generate_parameter_test_data():
    """Generate comprehensive test data for parameter validation."""
    print("Generating parameter test data...")
    
    # Create output directory
    Path("test_models").mkdir(exist_ok=True)
    
    # Run all tests
    test_feature_fraction()
    test_bagging()
    test_regularization()
    test_min_data_in_leaf()
    
    print("\nTest data generation complete!")
    print("Models saved in test_models/ directory")


if __name__ == "__main__":
    generate_parameter_test_data()