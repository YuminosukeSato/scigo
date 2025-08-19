#!/usr/bin/env python3
"""
Generate test data for Go/Python LightGBM compatibility testing.
This script trains LightGBM models in Python and saves both the models
and test predictions for verification in Go.
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer, load_iris, make_regression, make_classification
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Create output directory
OUTPUT_DIR = "testdata/compatibility"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_data(X, y, predictions, prefix, extra_meta=None):
    """Save test data and predictions to CSV files."""
    # Save test features
    pd.DataFrame(X).to_csv(
        os.path.join(OUTPUT_DIR, f"{prefix}_X_test.csv"),
        index=False,
        header=False
    )
    
    # Save test labels
    pd.DataFrame(y).to_csv(
        os.path.join(OUTPUT_DIR, f"{prefix}_y_test.csv"),
        index=False,
        header=False
    )
    
    # Save predictions
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(-1, 1)
    pd.DataFrame(predictions).to_csv(
        os.path.join(OUTPUT_DIR, f"{prefix}_predictions.csv"),
        index=False,
        header=False
    )

    # Save minimal JSON metadata for compatibility validation
    meta = {
        "task": prefix,
        "X_test_shape": list(np.shape(X)),
        "y_test_shape": list(np.shape(y)),
        "predictions_shape": list(np.shape(predictions)),
    }
    if isinstance(extra_meta, dict):
        meta.update(extra_meta)
    try:
        import json
        with open(os.path.join(OUTPUT_DIR, f"{prefix}_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

def save_dump_model(booster: lgb.Booster, prefix: str) -> None:
    """Save dump_model() output to JSON and a minimal per-tree structure for C-API parity tests."""
    import json
    dump = booster.dump_model()
    # Full dump model
    with open(os.path.join(OUTPUT_DIR, f"{prefix}_model.json"), "w") as f:
        json.dump(dump, f, indent=2)

    # Minimal JSON for C-API structure comparison (tree meta only)
    trees_min = []
    for ti, info in enumerate(dump.get("tree_info", [])):
        meta = {
            "tree_index": int(info.get("tree_index", ti)),
            "num_leaves": int(info.get("num_leaves", 0)),
            "num_nodes": int(info.get("num_leaves", 0)) - 1 if int(info.get("num_leaves", 0)) > 0 else 0,
            "shrinkage": float(info.get("shrinkage", 0.0)),
        }
        trees_min.append(meta)
    with open(os.path.join(OUTPUT_DIR, f"{prefix}_tree_structure.json"), "w") as f:
        json.dump(trees_min, f, indent=2)

def generate_regression_data():
    """Generate regression test case."""
    print("Generating regression test data...")
    
    # Use a synthetic dataset for reproducibility
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        noise=0.1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train LightGBM model
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
    )
    
    # Save model
    model.save_model(
        os.path.join(OUTPUT_DIR, "regression_model.txt"),
        num_iteration=model.best_iteration
    )
    save_dump_model(model, "regression")
    
    # Make predictions
    predictions = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Save data
    save_data(
        X_test,
        y_test,
        predictions,
        "regression",
        extra_meta={
            "objective": "regression",
            "num_trees": int(model.num_trees()),
            "best_iteration": int(model.best_iteration or 0),
            "params": params,
        },
    )
    
    print(f"  Regression model saved with {model.num_trees()} trees")
    print(f"  Test RMSE: {np.sqrt(np.mean((predictions - y_test)**2)):.4f}")

def generate_binary_classification_data():
    """Generate binary classification test case."""
    print("Generating binary classification test data...")
    
    # Use breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train LightGBM model
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
    )
    
    # Save model
    model.save_model(
        os.path.join(OUTPUT_DIR, "binary_model.txt"),
        num_iteration=model.best_iteration
    )
    save_dump_model(model, "binary")
    
    # Make predictions (probabilities)
    predictions = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Save data
    save_data(
        X_test,
        y_test,
        predictions,
        "binary",
        extra_meta={
            "objective": "binary",
            "num_trees": int(model.num_trees()),
            "best_iteration": int(model.best_iteration or 0),
            "params": params,
        },
    )
    
    # Also save probability predictions for both classes
    proba_both = np.column_stack([1 - predictions, predictions])
    pd.DataFrame(proba_both).to_csv(
        os.path.join(OUTPUT_DIR, "binary_probabilities.csv"),
        index=False,
        header=False
    )
    
    print(f"  Binary model saved with {model.num_trees()} trees")
    print(f"  Test Accuracy: {np.mean((predictions > 0.5) == y_test):.4f}")

def generate_multiclass_classification_data():
    """Generate multiclass classification test case."""
    print("Generating multiclass classification test data...")
    
    # Use iris dataset
    data = load_iris()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train LightGBM model
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
    )
    
    # Save model
    model.save_model(
        os.path.join(OUTPUT_DIR, "multiclass_model.txt"),
        num_iteration=model.best_iteration
    )
    save_dump_model(model, "multiclass")
    
    # Make predictions (probabilities for each class)
    predictions = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Save data
    save_data(
        X_test,
        y_test,
        predictions,
        "multiclass",
        extra_meta={
            "objective": "multiclass",
            "num_class": 3,
            "num_trees": int(model.num_trees()),
            "best_iteration": int(model.best_iteration or 0),
            "params": params,
        },
    )
    
    # Also save class predictions
    class_predictions = np.argmax(predictions, axis=1)
    pd.DataFrame(class_predictions).to_csv(
        os.path.join(OUTPUT_DIR, "multiclass_classes.csv"),
        index=False,
        header=False
    )
    
    print(f"  Multiclass model saved with {model.num_trees()} trees")
    print(f"  Test Accuracy: {np.mean(class_predictions == y_test):.4f}")

def generate_special_cases():
    """Generate models with special cases for testing."""
    print("Generating special case models...")
    
    # 1. Model with NaN handling
    X, y = make_regression(n_samples=500, n_features=5, random_state=42)
    # Introduce NaN values
    X[::10, 0] = np.nan
    X[::15, 2] = np.nan
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    params = {
        'objective': 'regression',
        'num_leaves': 15,
        'learning_rate': 0.1,
        'verbose': -1,
        'seed': 42,
        'use_missing': True,  # Enable missing value handling
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=50)
    
    model.save_model(os.path.join(OUTPUT_DIR, "nan_handling_model.txt"))
    save_dump_model(model, "nan_handling")
    predictions = model.predict(X_test)
    save_data(X_test, y_test, predictions, "nan_handling")
    
    # 2. Deep tree model
    X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    params = {
        'objective': 'regression',
        'num_leaves': 127,  # Large number of leaves
        'max_depth': 10,    # Deep trees
        'learning_rate': 0.05,
        'verbose': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=20)
    
    model.save_model(os.path.join(OUTPUT_DIR, "deep_tree_model.txt"))
    save_dump_model(model, "deep_tree")
    predictions = model.predict(X_test)
    save_data(X_test, y_test, predictions, "deep_tree")
    
    print("  Special case models generated")

def generate_feature_importance_test():
    """Generate a model with clear feature importance patterns."""
    print("Generating feature importance test model...")
    
    # Create data where feature importance is obvious
    np.random.seed(42)
    n_samples = 1000
    
    # Feature 0: Most important (directly affects target)
    X0 = np.random.randn(n_samples)
    # Feature 1: Somewhat important
    X1 = np.random.randn(n_samples) * 0.5
    # Feature 2: Not important (random noise)
    X2 = np.random.randn(n_samples)
    # Feature 3: Not important (constant)
    X3 = np.ones(n_samples) * 5
    
    X = np.column_stack([X0, X1, X2, X3])
    # Target strongly depends on X0, weakly on X1
    y = 2 * X0 + 0.5 * X1 + 0.1 * np.random.randn(n_samples)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    params = {
        'objective': 'regression',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'verbose': -1,
        'seed': 42,
        'min_data_in_leaf': 5
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=50)
    
    model.save_model(os.path.join(OUTPUT_DIR, "feature_importance_model.txt"))
    save_dump_model(model, "feature_importance")
    
    # Get feature importance
    importance_gain = model.feature_importance(importance_type='gain')
    importance_split = model.feature_importance(importance_type='split')
    
    # Save feature importance
    pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(4)],
        'gain': importance_gain,
        'split': importance_split
    }).to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
    
    predictions = model.predict(X_test)
    save_data(X_test, y_test, predictions, "feature_importance")
    
    print("  Feature importance model generated")
    print(f"  Importance (gain): {importance_gain}")

def main():
    """Generate all test data."""
    print("="*60)
    print("LightGBM Go/Python Compatibility Test Data Generator")
    print("="*60)
    
    # Check LightGBM version
    print(f"LightGBM version: {lgb.__version__}")
    print()
    
    # Generate test cases
    generate_regression_data()
    print()
    generate_binary_classification_data()
    print()
    generate_multiclass_classification_data()
    print()
    generate_special_cases()
    print()
    generate_feature_importance_test()
    
    print()
    print("="*60)
    print(f"Test data generated in: {OUTPUT_DIR}")
    print("Files created:")
    for file in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, file))
        print(f"  - {file:40} ({size:,} bytes)")
    print("="*60)

if __name__ == "__main__":
    main()
