#!/usr/bin/env python3
"""
LightGBMの段階的予測検証スクリプト
各ツリーの寄与を個別に確認し、Go実装との比較用データを生成
"""

import lightgbm as lgb
import numpy as np
import json

def main():
    # モデルとテストデータを読み込み
    model = lgb.Booster(model_file='testdata/compatibility/regression_model.txt')
    X_test = np.loadtxt('testdata/compatibility/regression_X_test.csv', delimiter=',')
    
    # 最初のサンプルを使用（Go側と同じ）
    sample = X_test[0:1]  # Shape: (1, 10)
    
    print("=== LightGBM Python 段階的予測検証 ===")
    print(f"Model trees: {model.num_trees()}")
    print(f"Best iteration: {model.best_iteration}")
    print(f"Sample features: {sample[0]}")
    print()
    
    # 最終予測値
    final_pred = model.predict(sample)[0]
    print(f"Final prediction: {final_pred}")
    print()
    
    # 段階的予測（1ツリーずつ追加）
    print("=== 段階的予測（1ツリーずつ） ===")
    cumulative_predictions = []
    
    for i in range(1, min(11, model.num_trees() + 1)):  # 最初の10ツリー
        pred = model.predict(sample, num_iteration=i)[0]
        cumulative_predictions.append(pred)
        print(f"Trees 1-{i:2d}: {pred:12.6f}")
    
    print()
    
    # さらに詳細な段階（10, 20, 50, 100ツリー）
    print("=== 主要な段階での予測値 ===")
    for num_trees in [10, 20, 50, 100]:
        if num_trees <= model.num_trees():
            pred = model.predict(sample, num_iteration=num_trees)[0]
            print(f"Trees 1-{num_trees:3d}: {pred:12.6f}")
    
    print()
    
    # モデル構造の詳細
    print("=== モデル構造詳細 ===")
    model_json = model.dump_model()
    
    print(f"Feature names: {model_json.get('feature_names', [])}")
    print(f"Objective: {model_json.get('objective', 'unknown')}")
    print()
    
    # 最初の3ツリーの詳細
    print("=== 最初の3ツリーの詳細 ===")
    for i in range(min(3, len(model_json['tree_info']))):
        tree_info = model_json['tree_info'][i]
        print(f"Tree {i}:")
        print(f"  num_leaves: {tree_info['num_leaves']}")
        print(f"  shrinkage: {tree_info['shrinkage']}")
        
        # 個別のツリー予測
        single_tree_pred = model.predict(sample, start_iteration=i, num_iteration=1)[0]
        print(f"  single tree prediction: {single_tree_pred:12.6f}")
        print()
    
    # 検証用データを保存
    verification_data = {
        'sample_features': sample[0].tolist(),
        'final_prediction': float(final_pred),
        'cumulative_predictions': [float(p) for p in cumulative_predictions],
        'model_info': {
            'num_trees': model.num_trees(),
            'best_iteration': model.best_iteration,
            'objective': model_json.get('objective', 'unknown')
        }
    }
    
    with open('testdata/python_verification.json', 'w') as f:
        json.dump(verification_data, f, indent=2)
    
    print("検証データを testdata/python_verification.json に保存しました")

if __name__ == "__main__":
    main()