#!/usr/bin/env python3
"""
Generate test expectations for Go LightGBM implementation compatibility testing.
This script creates models and predictions using Python LightGBM to serve as
ground truth for comparison with the Go implementation.
"""

import json
import os
import sys
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_iris, load_breast_cancer, make_regression, make_classification
from sklearn.model_selection import train_test_split
from pathlib import Path

def create_output_dir(base_dir="testdata/expectations"):
    """Create output directory for test data."""
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    return base_dir

def generate_iris_test():
    """Generate test case using Iris dataset (multiclass)."""
    print("Generating Iris dataset test case...")
    
    # Load data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    # Parameters
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 42,
        'verbose': -1,
        'deterministic': True
    }
    
    # Train model
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=20)
    
    # Save model
    output_dir = create_output_dir()
    model_path = f"{output_dir}/iris_model.json"
    model.save_model(model_path)
    
    # Dump model for detailed structure
    dump_path = f"{output_dir}/iris_model_dump.json"
    with open(dump_path, 'w') as f:
        json.dump(model.dump_model(), f, indent=2)
    
    # Generate predictions
    predictions = {
        'predict': model.predict(X_test, num_iteration=model.best_iteration).tolist(),
        'predict_proba': model.predict(X_test, num_iteration=model.best_iteration).tolist(),
        'raw_score': model.predict(X_test, num_iteration=model.best_iteration, 
                                  raw_score=True).tolist(),
        'leaf_index': model.predict(X_test, num_iteration=model.best_iteration,
                                   pred_leaf=True).tolist()
    }
    
    # Save test data and expectations
    test_case = {
        'dataset': 'iris',
        'params': params,
        'X_test': X_test.tolist(),
        'y_test': y_test.tolist(),
        'predictions': predictions,
        'model_info': {
            'num_trees': model.num_trees(),
            'num_features': model.num_feature(),
            'feature_names': [f'feature_{i}' for i in range(model.num_feature())]
        }
    }
    
    expectations_path = f"{output_dir}/iris_expectations.json"
    with open(expectations_path, 'w') as f:
        json.dump(test_case, f, indent=2)
    
    print(f"✓ Iris test case saved to {expectations_path}")
    return test_case

def generate_binary_test():
    """Generate test case for binary classification."""
    print("Generating binary classification test case...")
    
    # Load data
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, test_size=0.2, random_state=42
    )
    
    # Parameters
    params = {
        'objective': 'binary',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'min_data_in_leaf': 20,
        'seed': 42,
        'verbose': -1,
        'deterministic': True
    }
    
    # Train model
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=20)
    
    # Save model
    output_dir = create_output_dir()
    model_path = f"{output_dir}/binary_model.json"
    model.save_model(model_path)
    
    # Dump model for detailed structure
    dump_path = f"{output_dir}/binary_model_dump.json"
    with open(dump_path, 'w') as f:
        json.dump(model.dump_model(), f, indent=2)
    
    # Generate predictions
    predictions = {
        'predict': model.predict(X_test, num_iteration=model.best_iteration).tolist(),
        'raw_score': model.predict(X_test, num_iteration=model.best_iteration,
                                  raw_score=True).tolist(),
        'leaf_index': model.predict(X_test, num_iteration=model.best_iteration,
                                   pred_leaf=True).tolist()
    }
    
    # Save test data and expectations
    test_case = {
        'dataset': 'breast_cancer',
        'params': params,
        'X_test': X_test.tolist(),
        'y_test': y_test.tolist(),
        'predictions': predictions,
        'model_info': {
            'num_trees': model.num_trees(),
            'num_features': model.num_feature(),
            'feature_names': [f'feature_{i}' for i in range(model.num_feature())]
        }
    }
    
    expectations_path = f"{output_dir}/binary_expectations.json"
    with open(expectations_path, 'w') as f:
        json.dump(test_case, f, indent=2)
    
    print(f"✓ Binary test case saved to {expectations_path}")
    return test_case

def generate_regression_test():
    """Generate test case for regression."""
    print("Generating regression test case...")
    
    # Create synthetic regression data
    X, y = make_regression(n_samples=1000, n_features=10, n_informative=5,
                          noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Parameters
    params = {
        'objective': 'regression',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.0,
        'lambda_l2': 0.1,
        'min_gain_to_split': 0.01,
        'seed': 42,
        'verbose': -1,
        'deterministic': True
    }
    
    # Train model
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=20)
    
    # Save model
    output_dir = create_output_dir()
    model_path = f"{output_dir}/regression_model.json"
    model.save_model(model_path)
    
    # Dump model for detailed structure
    dump_path = f"{output_dir}/regression_model_dump.json"
    with open(dump_path, 'w') as f:
        json.dump(model.dump_model(), f, indent=2)
    
    # Generate predictions
    predictions = {
        'predict': model.predict(X_test, num_iteration=model.best_iteration).tolist(),
        'raw_score': model.predict(X_test, num_iteration=model.best_iteration,
                                  raw_score=True).tolist(),
        'leaf_index': model.predict(X_test, num_iteration=model.best_iteration,
                                   pred_leaf=True).tolist()
    }
    
    # Save test data and expectations
    test_case = {
        'dataset': 'synthetic_regression',
        'params': params,
        'X_test': X_test.tolist(),
        'y_test': y_test.tolist(),
        'predictions': predictions,
        'model_info': {
            'num_trees': model.num_trees(),
            'num_features': model.num_feature(),
            'feature_names': [f'feature_{i}' for i in range(model.num_feature())]
        }
    }
    
    expectations_path = f"{output_dir}/regression_expectations.json"
    with open(expectations_path, 'w') as f:
        json.dump(test_case, f, indent=2)
    
    print(f"✓ Regression test case saved to {expectations_path}")
    return test_case

def generate_categorical_test():
    """Generate test case with categorical features."""
    print("Generating categorical features test case...")
    
    # Create data with categorical features
    n_samples = 1000
    np.random.seed(42)
    
    # Mix of numerical and categorical features
    X_num = np.random.randn(n_samples, 3)
    X_cat1 = np.random.choice(['A', 'B', 'C', 'D'], size=n_samples)
    X_cat2 = np.random.choice(['X', 'Y', 'Z'], size=n_samples)
    
    # Encode categorical features
    cat1_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    cat2_map = {'X': 0, 'Y': 1, 'Z': 2}
    X_cat1_encoded = np.array([cat1_map[x] for x in X_cat1])
    X_cat2_encoded = np.array([cat2_map[x] for x in X_cat2])
    
    # Combine features
    X = np.column_stack([X_num, X_cat1_encoded.reshape(-1, 1), X_cat2_encoded.reshape(-1, 1)])
    
    # Create target (binary classification)
    y = (X[:, 0] + X[:, 3] * 0.5 + X[:, 4] * 0.3 + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Parameters with categorical features
    params = {
        'objective': 'binary',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'categorical_feature': [3, 4],  # Indices of categorical features
        'seed': 42,
        'verbose': -1,
        'deterministic': True
    }
    
    # Train model
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=[3, 4])
    model = lgb.train(params, train_data, num_boost_round=20)
    
    # Save model
    output_dir = create_output_dir()
    model_path = f"{output_dir}/categorical_model.json"
    model.save_model(model_path)
    
    # Dump model for detailed structure
    dump_path = f"{output_dir}/categorical_model_dump.json"
    with open(dump_path, 'w') as f:
        json.dump(model.dump_model(), f, indent=2)
    
    # Generate predictions
    predictions = {
        'predict': model.predict(X_test, num_iteration=model.best_iteration).tolist(),
        'raw_score': model.predict(X_test, num_iteration=model.best_iteration,
                                  raw_score=True).tolist(),
        'leaf_index': model.predict(X_test, num_iteration=model.best_iteration,
                                   pred_leaf=True).tolist()
    }
    
    # Save test data and expectations
    test_case = {
        'dataset': 'categorical_features',
        'params': params,
        'X_test': X_test.tolist(),
        'y_test': y_test.tolist(),
        'predictions': predictions,
        'model_info': {
            'num_trees': model.num_trees(),
            'num_features': model.num_feature(),
            'feature_names': ['num_0', 'num_1', 'num_2', 'cat_1', 'cat_2'],
            'categorical_features': [3, 4]
        },
        'categorical_mappings': {
            'feature_3': cat1_map,
            'feature_4': cat2_map
        }
    }
    
    expectations_path = f"{output_dir}/categorical_expectations.json"
    with open(expectations_path, 'w') as f:
        json.dump(test_case, f, indent=2)
    
    print(f"✓ Categorical test case saved to {expectations_path}")
    return test_case

def main():
    """Main function to generate all test cases."""
    parser = argparse.ArgumentParser(description='Generate LightGBM test expectations')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['all', 'iris', 'binary', 'regression', 'categorical'],
                       help='Which dataset to generate expectations for')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LightGBM Test Expectations Generator")
    print("=" * 60)
    print(f"LightGBM version: {lgb.__version__}")
    print()
    
    if args.dataset == 'all':
        generate_iris_test()
        generate_binary_test()
        generate_regression_test()
        generate_categorical_test()
    elif args.dataset == 'iris':
        generate_iris_test()
    elif args.dataset == 'binary':
        generate_binary_test()
    elif args.dataset == 'regression':
        generate_regression_test()
    elif args.dataset == 'categorical':
        generate_categorical_test()
    
    print()
    print("✅ All test expectations generated successfully!")
    print(f"Output directory: {create_output_dir()}")

if __name__ == "__main__":
    main()