#!/usr/bin/env python3
"""
Create test LightGBM models in Python for compatibility testing with Go implementation.
This ensures numerical precision matches between Python and Go.
"""

import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split
import json
import os

# Output directory configuration
OUTPUT_DIR = 'testdata/compatibility'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_binary_classification_model():
    """Create a binary classification model using Iris dataset (2 classes)."""
    print("Creating binary classification model...")
    
    # Load Iris dataset and use only 2 classes
    iris = load_iris()
    X = iris.data[iris.target != 2]  # Remove class 2
    y = iris.target[iris.target != 2]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'verbose': -1,
        'seed': 42,
        'deterministic': True,
        'force_col_wise': True,  # For deterministic results
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=10)
    
    # Save model
    model.save_model(os.path.join(OUTPUT_DIR, 'binary_model.txt'))
    
    # Save model as JSON for inspection
    with open(os.path.join(OUTPUT_DIR, 'binary_model.json'), 'w') as f:
        json.dump(model.dump_model(), f, indent=2)
    
    # Make predictions on test samples for verification
    test_samples = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [6.2, 3.4, 5.4, 2.3],  # This should be from the removed class, but let's test
    ])
    
    predictions = model.predict(test_samples)
    
    print("Binary classification test predictions:")
    for i, pred in enumerate(predictions):
        print(f"  Sample {i}: {pred:.10f}")
    
    # Save test data and expected outputs
    test_data = {
        'model_file': 'binary_model.txt',
        'test_samples': test_samples.tolist(),
        'expected_predictions': predictions.tolist(),
        'model_params': params,
    }
    
    with open(os.path.join(OUTPUT_DIR, 'binary_test_data.json'), 'w') as f:
        json.dump(test_data, f, indent=2)
    
    return model, test_samples, predictions

def create_regression_model():
    """Create a regression model."""
    print("\nCreating regression model...")
    
    # Create synthetic regression data
    X, y = make_regression(
        n_samples=100,
        n_features=3,
        n_informative=3,
        noise=0.1,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    params = {
        'objective': 'regression',
        'metric': 'l2',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'verbose': -1,
        'seed': 42,
        'deterministic': True,
        'force_col_wise': True,
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=10)
    
    # Save model
    model.save_model(os.path.join(OUTPUT_DIR, 'regression_model.txt'))
    
    # Save model as JSON
    with open(os.path.join(OUTPUT_DIR, 'regression_model.json'), 'w') as f:
        json.dump(model.dump_model(), f, indent=2)
    
    # Make predictions on test samples
    test_samples = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])
    
    predictions = model.predict(test_samples)
    
    print("Regression test predictions:")
    for i, pred in enumerate(predictions):
        print(f"  Sample {i}: {pred:.10f}")
    
    # Save test data
    test_data = {
        'model_file': 'regression_model.txt',
        'test_samples': test_samples.tolist(),
        'expected_predictions': predictions.tolist(),
        'model_params': params,
    }
    
    with open(os.path.join(OUTPUT_DIR, 'regression_test_data.json'), 'w') as f:
        json.dump(test_data, f, indent=2)
    
    return model, test_samples, predictions

def create_multiclass_model():
    """Create a multiclass classification model."""
    print("\nCreating multiclass classification model...")
    
    # Load full Iris dataset (3 classes)
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'verbose': -1,
        'seed': 42,
        'deterministic': True,
        'force_col_wise': True,
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=10)
    
    # Save model
    model.save_model(os.path.join(OUTPUT_DIR, 'multiclass_model.txt'))
    
    # Save model as JSON
    with open(os.path.join(OUTPUT_DIR, 'multiclass_model.json'), 'w') as f:
        json.dump(model.dump_model(), f, indent=2)
    
    # Make predictions on test samples
    test_samples = np.array([
        [5.1, 3.5, 1.4, 0.2],  # Should be class 0 (setosa)
        [5.9, 3.0, 4.2, 1.5],  # Should be class 1 (versicolor)
        [6.3, 3.3, 6.0, 2.5],  # Should be class 2 (virginica)
    ])
    
    predictions = model.predict(test_samples)
    
    print("Multiclass test predictions:")
    for i, pred in enumerate(predictions):
        print(f"  Sample {i}: {pred}")
    
    # Save test data
    test_data = {
        'model_file': 'multiclass_model.txt',
        'test_samples': test_samples.tolist(),
        'expected_predictions': predictions.tolist(),
        'model_params': params,
    }
    
    with open(os.path.join(OUTPUT_DIR, 'multiclass_test_data.json'), 'w') as f:
        json.dump(test_data, f, indent=2)
    
    return model, test_samples, predictions

def verify_model_loading():
    """Verify that saved models can be loaded correctly."""
    print("\nVerifying model loading...")
    
    # Load and test binary model
    binary_model = lgb.Booster(model_file=os.path.join(OUTPUT_DIR, 'binary_model.txt'))
    with open(os.path.join(OUTPUT_DIR, 'binary_test_data.json'), 'r') as f:
        binary_data = json.load(f)
    
    binary_preds = binary_model.predict(np.array(binary_data['test_samples']))
    print("Binary model verification:")
    for i, (pred, expected) in enumerate(zip(binary_preds, binary_data['expected_predictions'])):
        diff = abs(pred - expected)
        print(f"  Sample {i}: pred={pred:.10f}, expected={expected:.10f}, diff={diff:.2e}")
    
    # Load and test regression model
    reg_model = lgb.Booster(model_file=os.path.join(OUTPUT_DIR, 'regression_model.txt'))
    with open(os.path.join(OUTPUT_DIR, 'regression_test_data.json'), 'r') as f:
        reg_data = json.load(f)
    
    reg_preds = reg_model.predict(np.array(reg_data['test_samples']))
    print("\nRegression model verification:")
    for i, (pred, expected) in enumerate(zip(reg_preds, reg_data['expected_predictions'])):
        diff = abs(pred - expected)
        print(f"  Sample {i}: pred={pred:.10f}, expected={expected:.10f}, diff={diff:.2e}")

def main():
    """Create all test models."""
    print("Creating LightGBM test models for Go compatibility testing")
    print("=" * 60)
    
    # Check LightGBM version
    print(f"LightGBM version: {lgb.__version__}")
    print(f"NumPy version: {np.__version__}")
    print()
    
    # Create models
    create_binary_classification_model()
    create_regression_model()
    create_multiclass_model()
    
    # Verify
    verify_model_loading()
    
    print("\n" + "=" * 60)
    print("Test models created successfully!")
    print(f"Files created in {OUTPUT_DIR} directory:")
    print("  - binary_model.txt/json")
    print("  - regression_model.txt/json")
    print("  - multiclass_model.txt/json")
    print("  - *_test_data.json (contains test samples and expected outputs)")
    print("\nRun 'go test ./sklearn/lightgbm' to test Go implementation compatibility")

if __name__ == "__main__":
    main()