#!/usr/bin/env python3
"""
Generate golden data for LightGBM C API compatibility testing.
This script creates reference data using Python LightGBM for exact numerical comparison.
"""

import json
import numpy as np
import lightgbm as lgb
from pathlib import Path
import struct

# Fix random seed for reproducibility
np.random.seed(42)

def save_array_binary(arr, filepath):
    """Save array in binary format for exact bit-level comparison."""
    with open(filepath, 'wb') as f:
        # Write shape information
        f.write(struct.pack('I', len(arr.shape)))
        for dim in arr.shape:
            f.write(struct.pack('I', dim))
        # Write data type (0=float32, 1=float64, 2=int32)
        if arr.dtype == np.float32:
            f.write(struct.pack('I', 0))
        elif arr.dtype == np.float64:
            f.write(struct.pack('I', 1))
        else:  # int32
            f.write(struct.pack('I', 2))
        # Write raw data
        arr.tofile(f)

def generate_minimal_regression_data():
    """Generate minimal test case: 20 samples, 3 features."""
    print("Generating minimal regression test case...")
    
    # Create more samples to allow splitting
    X = np.array([
        [1.0, 2.0, 0.5],
        [2.0, 3.0, 1.0],
        [3.0, 1.0, 1.5],
        [4.0, 2.0, 0.5],
        [5.0, 3.0, 1.0],
        [1.5, 2.5, 0.8],
        [2.5, 1.5, 1.2],
        [3.5, 2.8, 0.6],
        [4.5, 1.2, 1.4],
        [0.5, 3.5, 0.9],
        [1.2, 1.8, 1.1],
        [2.8, 2.2, 0.7],
        [3.2, 3.1, 1.3],
        [4.1, 0.9, 0.4],
        [0.8, 2.7, 1.6],
        [2.3, 1.4, 0.3],
        [3.7, 2.6, 1.7],
        [1.9, 3.3, 0.2],
        [4.6, 1.7, 1.8],
        [0.3, 2.9, 0.1]
    ], dtype=np.float32)
    
    # Simple linear relationship with deterministic small noise
    noise = np.array([0.05, -0.03, 0.02, -0.04, 0.01, 0.03, -0.02, 0.04, -0.01, 0.02,
                      -0.03, 0.01, -0.02, 0.03, -0.04, 0.02, -0.01, 0.04, -0.03, 0.01], dtype=np.float32)
    y = X[:, 0] * 0.5 + X[:, 1] * 0.3 - X[:, 2] * 0.2 + noise
    
    # Minimal parameters for reproducibility
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 3,  # Allow minimal splitting
        'max_depth': 2,
        'learning_rate': 0.1,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'bagging_freq': 0,
        'lambda_l2': 0.0,
        'lambda_l1': 0.0,
        'min_data_in_leaf': 2,
        'min_gain_to_split': 0.0,
        'verbosity': 2,
        'num_threads': 1,
        'seed': 42,
        'deterministic': True,
        'force_row_wise': True,
        'min_data_in_bin': 3  # Ensure enough data for binning
    }
    
    # Create dataset
    train_data = lgb.Dataset(X, label=y)
    
    # Initialize booster
    booster = lgb.Booster(params=params, train_set=train_data)
    
    # Track intermediate values for each iteration
    intermediate_data = {
        'X': X.tolist(),
        'y': y.tolist(),
        'params': params,
        'iterations': []
    }
    
    # Perform 3 iterations and record state
    for i in range(3):
        print(f"  Iteration {i+1}...")
        
        # Update one iteration
        booster.update()
        
        # Get current predictions
        predictions = booster.predict(X, num_iteration=i+1)
        
        # Calculate residuals (gradients for squared loss)
        residuals = y - predictions
        gradients = -2.0 * residuals  # For squared loss
        hessians = np.full_like(gradients, 2.0)  # Constant for squared loss
        
        # Get tree structure (for debugging)
        tree_dump = booster.dump_model(num_iteration=i+1)
        
        iteration_data = {
            'iteration': i + 1,
            'predictions': predictions.tolist(),
            'residuals': residuals.tolist(),
            'gradients': gradients.tolist(),
            'hessians': hessians.tolist(),
            'tree_count': tree_dump['tree_info'][-1]['tree_index'] + 1 if tree_dump['tree_info'] else 0,
            'feature_importance': booster.feature_importance(importance_type='split').tolist()
        }
        
        intermediate_data['iterations'].append(iteration_data)
    
    # Save final model and predictions
    final_predictions = booster.predict(X)
    intermediate_data['final_predictions'] = final_predictions.tolist()
    
    # Create output directory
    output_dir = Path('tests/compatibility/golden_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all data
    with open(output_dir / 'minimal_regression.json', 'w') as f:
        json.dump(intermediate_data, f, indent=2)
    
    # Save binary arrays for exact comparison
    save_array_binary(X, output_dir / 'minimal_regression_X.bin')
    save_array_binary(y, output_dir / 'minimal_regression_y.bin')
    save_array_binary(final_predictions, output_dir / 'minimal_regression_predictions.bin')
    
    # Save model in text format for inspection
    booster.save_model(str(output_dir / 'minimal_regression_model.txt'))
    
    # Also save in JSON format for structure analysis
    model_json = booster.dump_model()
    with open(output_dir / 'minimal_regression_model.json', 'w') as f:
        json.dump(model_json, f, indent=2)
    
    print(f"  Saved golden data to {output_dir}")
    return intermediate_data

def verify_reproducibility():
    """Verify that the generation is reproducible."""
    print("\nVerifying reproducibility...")
    
    # Generate twice and compare
    data1 = generate_minimal_regression_data()
    data2 = generate_minimal_regression_data()
    
    # Check if predictions match exactly
    for i, (iter1, iter2) in enumerate(zip(data1['iterations'], data2['iterations'])):
        pred1 = np.array(iter1['predictions'])
        pred2 = np.array(iter2['predictions'])
        if not np.array_equal(pred1, pred2):
            print(f"  WARNING: Iteration {i+1} predictions don't match exactly!")
            print(f"    Max difference: {np.max(np.abs(pred1 - pred2))}")
        else:
            print(f"  ✓ Iteration {i+1} predictions match exactly")
    
    final1 = np.array(data1['final_predictions'])
    final2 = np.array(data2['final_predictions'])
    if np.array_equal(final1, final2):
        print("  ✓ Final predictions are reproducible")
    else:
        print(f"  WARNING: Final predictions differ by max {np.max(np.abs(final1 - final2))}")

def main():
    print("=== LightGBM Golden Data Generator ===\n")
    
    # Generate minimal test case
    generate_minimal_regression_data()
    
    # Verify reproducibility
    verify_reproducibility()
    
    print("\n✓ Golden data generation complete!")

if __name__ == "__main__":
    main()