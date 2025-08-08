#!/usr/bin/env python3
"""
Train LightGBM models with Python for baseline comparison.
Ensures deterministic results for verification against Go implementation.
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_iris, make_regression, make_classification
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Output directories
OUTPUT_DIR = "../testdata"
REGRESSION_DIR = f"{OUTPUT_DIR}/regression"
BINARY_DIR = f"{OUTPUT_DIR}/binary"
MULTICLASS_DIR = f"{OUTPUT_DIR}/multiclass"

# Create directories
for dir_path in [REGRESSION_DIR, BINARY_DIR, MULTICLASS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

def save_data(X, y, prefix, output_dir):
    """Save training and test data to CSV files."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Save as CSV
    pd.DataFrame(X_train).to_csv(f"{output_dir}/{prefix}_X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(f"{output_dir}/{prefix}_X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv(f"{output_dir}/{prefix}_y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv(f"{output_dir}/{prefix}_y_test.csv", index=False)
    
    return X_train, X_test, y_train, y_test

def train_regression_model():
    """Train a regression model."""
    print("Training regression model...")
    
    # Generate regression data
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        noise=0.1,
        random_state=RANDOM_STATE
    )
    
    X_train, X_test, y_train, y_test = save_data(X, y, "regression", REGRESSION_DIR)
    
    # Create dataset with deterministic parameters
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Parameters for deterministic training
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 1.0,  # No feature sampling
        'bagging_fraction': 1.0,  # No data sampling
        'bagging_freq': 0,
        'max_depth': 5,
        'min_data_in_leaf': 20,
        'lambda_l2': 1.0,
        'lambda_l1': 0.0,
        'min_gain_to_split': 0.0,
        'verbosity': 1,
        'seed': RANDOM_STATE,
        'deterministic': True,
        'force_row_wise': True,
        'num_threads': 1
    }
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=10,
        valid_sets=[train_data],
        callbacks=[lgb.log_evaluation(1)]
    )
    
    # Save model
    model.save_model(f"{REGRESSION_DIR}/model.txt")
    
    # Save predictions
    train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    pd.DataFrame({'prediction': train_pred}).to_csv(
        f"{REGRESSION_DIR}/train_predictions.csv", index=False
    )
    pd.DataFrame({'prediction': test_pred}).to_csv(
        f"{REGRESSION_DIR}/test_predictions.csv", index=False
    )
    
    # Save JSON format for Go tests
    test_data = {
        'X_test': X_test.tolist(),
        'y_test': y_test.tolist(),
        'predictions': test_pred.tolist()
    }
    with open(f"{REGRESSION_DIR}/test_data.json", 'w') as f:
        json.dump(test_data, f)
    
    # Save parameters
    with open(f"{REGRESSION_DIR}/params.json", 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"Regression model saved to {REGRESSION_DIR}/")
    return model

def train_binary_model():
    """Train a binary classification model."""
    print("Training binary classification model...")
    
    # Generate binary classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=RANDOM_STATE
    )
    
    X_train, X_test, y_train, y_test = save_data(X, y, "binary", BINARY_DIR)
    
    # Create dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Parameters for deterministic training
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'bagging_freq': 0,
        'max_depth': 5,
        'min_data_in_leaf': 20,
        'lambda_l2': 1.0,
        'lambda_l1': 0.0,
        'min_gain_to_split': 0.0,
        'verbosity': 1,
        'seed': RANDOM_STATE,
        'deterministic': True,
        'force_row_wise': True,
        'num_threads': 1
    }
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=10,
        valid_sets=[train_data],
        callbacks=[lgb.log_evaluation(1)]
    )
    
    # Save model
    model.save_model(f"{BINARY_DIR}/model.txt")
    
    # Save predictions (probabilities)
    train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    pd.DataFrame({'probability': train_pred}).to_csv(
        f"{BINARY_DIR}/train_predictions.csv", index=False
    )
    pd.DataFrame({'probability': test_pred}).to_csv(
        f"{BINARY_DIR}/test_predictions.csv", index=False
    )
    
    # Save JSON format for Go tests
    # Create 2D array for binary probabilities [prob_class_0, prob_class_1]
    test_proba = np.column_stack([1 - test_pred, test_pred])
    test_data = {
        'X_test': X_test.tolist(),
        'y_test': y_test.tolist(),
        'predictions': (test_pred > 0.5).astype(int).tolist(),
        'predict_proba': test_proba.tolist()
    }
    with open(f"{BINARY_DIR}/test_data.json", 'w') as f:
        json.dump(test_data, f)
    
    # Save parameters
    with open(f"{BINARY_DIR}/params.json", 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"Binary model saved to {BINARY_DIR}/")
    return model

def train_multiclass_model():
    """Train a multiclass classification model."""
    print("Training multiclass classification model...")
    
    # Use Iris dataset for multiclass
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Add more synthetic samples
    n_copies = 10
    X = np.vstack([X] * n_copies)
    y = np.hstack([y] * n_copies)
    
    # Add small noise to make it more realistic
    X += np.random.RandomState(RANDOM_STATE).normal(0, 0.01, X.shape)
    
    X_train, X_test, y_train, y_test = save_data(X, y, "multiclass", MULTICLASS_DIR)
    
    # Create dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Parameters for deterministic training
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'bagging_freq': 0,
        'max_depth': 5,
        'min_data_in_leaf': 20,
        'lambda_l2': 1.0,
        'lambda_l1': 0.0,
        'min_gain_to_split': 0.0,
        'verbosity': 1,
        'seed': RANDOM_STATE,
        'deterministic': True,
        'force_row_wise': True,
        'num_threads': 1
    }
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=10,
        valid_sets=[train_data],
        callbacks=[lgb.log_evaluation(1)]
    )
    
    # Save model
    model.save_model(f"{MULTICLASS_DIR}/model.txt")
    
    # Save predictions (probabilities for each class)
    train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Save as DataFrame with column names
    train_pred_df = pd.DataFrame(train_pred, columns=[f'class_{i}' for i in range(3)])
    test_pred_df = pd.DataFrame(test_pred, columns=[f'class_{i}' for i in range(3)])
    
    train_pred_df.to_csv(f"{MULTICLASS_DIR}/train_predictions.csv", index=False)
    test_pred_df.to_csv(f"{MULTICLASS_DIR}/test_predictions.csv", index=False)
    
    # Save JSON format for Go tests
    test_data = {
        'X_test': X_test.tolist(),
        'y_test': y_test.tolist(),
        'predictions': np.argmax(test_pred, axis=1).tolist(),
        'predict_proba': test_pred.tolist()
    }
    with open(f"{MULTICLASS_DIR}/test_data.json", 'w') as f:
        json.dump(test_data, f)
    
    # Save parameters
    with open(f"{MULTICLASS_DIR}/params.json", 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"Multiclass model saved to {MULTICLASS_DIR}/")
    return model

def save_model_structure(model, output_path):
    """Save detailed model structure for comparison."""
    # Dump model to JSON
    model_json = model.dump_model()
    
    with open(output_path, 'w') as f:
        json.dump(model_json, f, indent=2)
    
    # Extract tree information
    trees_info = []
    for tree in model_json['tree_info']:
        tree_structure = tree['tree_structure']
        trees_info.append({
            'num_leaves': tree['num_leaves'],
            'num_nodes': count_nodes(tree_structure),
            'max_depth': get_max_depth(tree_structure)
        })
    
    return trees_info

def count_nodes(node):
    """Count total nodes in a tree."""
    if 'leaf_value' in node:
        return 1
    return 1 + count_nodes(node['left_child']) + count_nodes(node['right_child'])

def get_max_depth(node, depth=0):
    """Get maximum depth of a tree."""
    if 'leaf_value' in node:
        return depth
    left_depth = get_max_depth(node['left_child'], depth + 1)
    right_depth = get_max_depth(node['right_child'], depth + 1)
    return max(left_depth, right_depth)

def main():
    """Train all models."""
    print("Starting LightGBM model training for baseline comparison...")
    print(f"Random seed: {RANDOM_STATE}")
    print("=" * 50)
    
    # Train models
    reg_model = train_regression_model()
    save_model_structure(reg_model, f"{REGRESSION_DIR}/model_structure.json")
    print("=" * 50)
    
    bin_model = train_binary_model()
    save_model_structure(bin_model, f"{BINARY_DIR}/model_structure.json")
    print("=" * 50)
    
    multi_model = train_multiclass_model()
    save_model_structure(multi_model, f"{MULTICLASS_DIR}/model_structure.json")
    print("=" * 50)
    
    print("\nAll models trained successfully!")
    print("Models and data saved to testdata/")
    print("\nNext steps:")
    print("1. Run generate_baseline.py to generate detailed baselines")
    print("2. Run comparison tests to verify Go implementation")

if __name__ == "__main__":
    main()