#!/usr/bin/env python3
"""
Generate precision test data for Go LightGBM implementation.
This script creates detailed traces of prediction process including intermediate values.
"""

import json
import numpy as np
import lightgbm as lgb
from pathlib import Path


def sigmoid(x):
    """Stable sigmoid implementation."""
    if x >= 0:
        exp_neg_x = np.exp(-x)
        return 1.0 / (1.0 + exp_neg_x)
    else:
        exp_x = np.exp(x)
        return exp_x / (1.0 + exp_x)


def trace_prediction(model, X, verbose=True):
    """
    Trace the prediction process step by step.
    Returns detailed information about each tree's contribution.
    """
    # Get model dump for tree structure
    model_dump = model.dump_model()
    
    # Initialize trace data
    trace = {
        "input": X.tolist() if isinstance(X, np.ndarray) else X,
        "num_trees": model.num_trees(),
        "num_classes": model_dump.get("num_class", 1),
        "objective": model_dump.get("objective", "regression"),
        "trees": [],
        "cumulative_scores": [],
        "final_raw_score": None,
        "final_prediction": None
    }
    
    # Get leaf indices for tracing
    leaf_indices = model.predict(X.reshape(1, -1), pred_leaf=True)[0]
    
    # Get raw scores (before transformation)
    raw_scores = model.predict(X.reshape(1, -1), raw_score=True)[0]
    
    # Get final prediction (after transformation)
    final_pred = model.predict(X.reshape(1, -1))[0]
    
    # For binary classification, track single score
    if trace["num_classes"] <= 2:
        cumulative_score = 0.0
    else:
        cumulative_score = np.zeros(trace["num_classes"])
    
    # Trace each tree's contribution
    for tree_idx in range(model.num_trees()):
        tree_info = model_dump["tree_info"][tree_idx]
        tree_structure = tree_info["tree_structure"]
        
        # Get leaf value for this tree
        leaf_idx = leaf_indices[tree_idx]
        leaf_value = get_leaf_value(tree_structure, leaf_idx)
        
        # Apply shrinkage
        shrinkage = tree_info.get("shrinkage", 1.0)
        tree_output = leaf_value * shrinkage
        
        # Update cumulative score
        if trace["num_classes"] <= 2:
            cumulative_score += tree_output
            current_cumulative = float(cumulative_score)
        else:
            class_idx = tree_idx % trace["num_classes"]
            cumulative_score[class_idx] += tree_output
            current_cumulative = cumulative_score.copy().tolist()
        
        # Record tree trace
        tree_trace = {
            "tree_index": tree_idx,
            "leaf_index": int(leaf_idx),
            "leaf_value": float(leaf_value),
            "shrinkage": float(shrinkage),
            "tree_output": float(tree_output),
            "cumulative_score": current_cumulative
        }
        
        trace["trees"].append(tree_trace)
        trace["cumulative_scores"].append(current_cumulative)
        
        if verbose:
            print(f"Tree {tree_idx}: leaf_idx={leaf_idx}, "
                  f"leaf_value={leaf_value:.10f}, "
                  f"output={tree_output:.10f}, "
                  f"cumulative={current_cumulative}")
    
    # Record final values
    trace["final_raw_score"] = raw_scores.tolist() if isinstance(raw_scores, np.ndarray) else float(raw_scores)
    trace["final_prediction"] = final_pred.tolist() if isinstance(final_pred, np.ndarray) else float(final_pred)
    
    # Apply objective transformation for verification
    if "binary" in trace["objective"].lower():
        trace["expected_prediction"] = float(sigmoid(cumulative_score)) if trace["num_classes"] <= 2 else None
    else:
        if isinstance(cumulative_score, np.ndarray):
            trace["expected_prediction"] = cumulative_score.tolist()
        else:
            trace["expected_prediction"] = float(cumulative_score)
    
    return trace


def get_leaf_value(tree_structure, leaf_idx):
    """
    Extract leaf value from tree structure given leaf index.
    """
    # Count leaves to find the target
    leaf_counter = [0]
    
    def traverse(node):
        if "leaf_index" in node:
            # This is a leaf
            if leaf_counter[0] == leaf_idx:
                return node["leaf_value"]
            leaf_counter[0] += 1
            return None
        else:
            # Internal node
            if "left_child" in node:
                result = traverse(node["left_child"])
                if result is not None:
                    return result
            if "right_child" in node:
                result = traverse(node["right_child"])
                if result is not None:
                    return result
            return None
    
    # Handle simple leaf-only tree
    if "leaf_value" in tree_structure and "split_index" not in tree_structure:
        return tree_structure["leaf_value"]
    
    result = traverse(tree_structure)
    return result if result is not None else 0.0


def create_test_cases():
    """Create various test cases for precision testing."""
    test_cases = []
    
    # Test Case 1: Simple binary classification
    print("Creating Test Case 1: Binary classification...")
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 1, 1, 0])
    
    params = {
        "objective": "binary",
        "num_leaves": 3,
        "min_data_in_leaf": 1,
        "learning_rate": 0.1,
        "num_iterations": 5,
        "verbose": -1,
        "seed": 42,
        "force_col_wise": True,
        "deterministic": True
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data)
    
    # Test points
    test_points = [
        [0.3, 0.7],
        [0.8, 0.2],
        [0.5, 0.5]
    ]
    
    for point in test_points:
        trace = trace_prediction(model, np.array(point), verbose=False)
        test_cases.append({
            "name": f"binary_classification_point_{point}",
            "model_file": "test_models/binary_simple.json",
            "trace": trace
        })
    
    # Save model
    Path("test_models").mkdir(exist_ok=True)
    model.save_model("test_models/binary_simple.json")
    
    # Test Case 2: Regression
    print("Creating Test Case 2: Regression...")
    X_train = np.random.randn(20, 3)
    y_train = X_train[:, 0] * 2 + X_train[:, 1] - X_train[:, 2] + np.random.randn(20) * 0.1
    
    params = {
        "objective": "regression",
        "num_leaves": 5,
        "learning_rate": 0.1,
        "num_iterations": 10,
        "verbose": -1,
        "seed": 42,
        "force_col_wise": True,
        "deterministic": True
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data)
    
    # Test points
    test_points = [
        [0.0, 0.0, 0.0],
        [1.0, -0.5, 0.3],
        [-0.8, 1.2, -0.4]
    ]
    
    for point in test_points:
        trace = trace_prediction(model, np.array(point), verbose=False)
        test_cases.append({
            "name": f"regression_point_{point}",
            "model_file": "test_models/regression_simple.json",
            "trace": trace
        })
    
    model.save_model("test_models/regression_simple.json")
    
    # Test Case 3: Multiclass classification
    print("Creating Test Case 3: Multiclass classification...")
    X_train = np.random.randn(30, 4)
    y_train = np.random.randint(0, 3, 30)
    
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "num_leaves": 4,
        "learning_rate": 0.1,
        "num_iterations": 6,  # 2 iterations per class
        "verbose": -1,
        "seed": 42,
        "force_col_wise": True,
        "deterministic": True
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data)
    
    # Test points
    test_points = [
        [0.5, -0.5, 0.5, -0.5],
        [1.0, 1.0, -1.0, -1.0]
    ]
    
    for point in test_points:
        trace = trace_prediction(model, np.array(point), verbose=False)
        test_cases.append({
            "name": f"multiclass_point_{point}",
            "model_file": "test_models/multiclass_simple.json",
            "trace": trace
        })
    
    model.save_model("test_models/multiclass_simple.json")
    
    return test_cases


def main():
    """Generate precision test data."""
    print("Generating precision test data...")
    
    # Create test cases
    test_cases = create_test_cases()
    
    # Save all test cases
    output = {
        "description": "Precision test data for Go LightGBM implementation",
        "test_cases": test_cases
    }
    
    with open("precision_test_data.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Generated {len(test_cases)} test cases")
    print("Saved to precision_test_data.json")
    
    # Print summary
    print("\nTest case summary:")
    for tc in test_cases:
        trace = tc["trace"]
        print(f"  {tc['name']}:")
        print(f"    Trees: {trace['num_trees']}")
        print(f"    Final raw score: {trace['final_raw_score']}")
        print(f"    Final prediction: {trace['final_prediction']}")


if __name__ == "__main__":
    main()