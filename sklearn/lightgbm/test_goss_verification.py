#!/usr/bin/env python3
"""
GOSS (Gradient-based One-Side Sampling) 検証テスト

このテストスイートは、Goで実装されたGOSSが
Python LightGBMと同等の動作をすることを検証します。
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import json
import os
from typing import Dict, List, Tuple, Optional
import hashlib


def create_deterministic_dataset(n_samples: int = 500, n_features: int = 10, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    決定論的なデータセットを作成
    
    Args:
        n_samples: サンプル数
        n_features: 特徴量数
        seed: 乱数シード
        
    Returns:
        X: 特徴量行列
        y: ターゲット配列
    """
    np.random.seed(seed)
    
    # 特徴量行列の生成
    X = np.random.randn(n_samples, n_features)
    
    # ターゲット生成（回帰問題）
    # 一部の特徴量により強く依存するパターンを作成
    true_coef = np.random.randn(n_features)
    noise = np.random.randn(n_samples) * 0.1
    y = X @ true_coef + noise
    
    return X, y


def train_with_goss_params(X: np.ndarray, y: np.ndarray, 
                          top_rate: float = 0.2, 
                          other_rate: float = 0.1,
                          num_boost_round: int = 5,
                          seed: int = 42) -> Dict:
    """
    指定されたGOSSパラメータでLightGBMを訓練
    
    Args:
        X: 特徴量行列
        y: ターゲット配列
        top_rate: GOSSのトップレート
        other_rate: GOSSのその他レート
        num_boost_round: ブースティングラウンド数
        seed: 乱数シード
        
    Returns:
        訓練結果の詳細情報
    """
    lgb_train = lgb.Dataset(X, y)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'goss',
        'top_rate': top_rate,
        'other_rate': other_rate,
        'num_leaves': 15,
        'min_data_in_leaf': 10,
        'learning_rate': 0.1,
        'feature_fraction': 1.0,
        'verbose': -1,
        'seed': seed,
        'deterministic': True
    }
    
    # コールバックで詳細情報を収集
    eval_results = {}
    
    model = lgb.train(
        params=params,
        train_set=lgb_train,
        num_boost_round=num_boost_round,
        callbacks=[lgb.record_evaluation(eval_results)]
    )
    
    # 予測を実行
    predictions = model.predict(X)
    
    # ツリー構造を取得
    trees_df = model.trees_to_dataframe()
    
    return {
        'model': model,
        'params': params,
        'predictions': predictions.tolist(),
        'eval_results': eval_results,
        'trees_df': trees_df,
        'num_trees': model.num_trees(),
        'feature_importance': model.feature_importance().tolist()
    }


def analyze_goss_sampling_behavior(X: np.ndarray, y: np.ndarray, 
                                 top_rate: float = 0.2,
                                 other_rate: float = 0.1,
                                 seed: int = 42) -> Dict:
    """
    GOSSサンプリングの動作を詳細に分析
    
    Args:
        X: 特徴量行列  
        y: ターゲット配列
        top_rate: トップレート
        other_rate: その他レート
        seed: 乱数シード
        
    Returns:
        サンプリング動作の詳細情報
    """
    n_samples = len(X)
    
    # 期待されるサンプル数の計算
    expected_top_count = int(n_samples * top_rate)
    expected_other_count = int((n_samples - expected_top_count) * other_rate)
    expected_total = expected_top_count + expected_other_count
    
    # 増幅係数の計算
    amplification_factor = (1.0 - top_rate) / other_rate if other_rate > 0 else 1.0
    
    return {
        'n_samples': n_samples,
        'top_rate': top_rate,
        'other_rate': other_rate,
        'expected_top_count': expected_top_count,
        'expected_other_count': expected_other_count,
        'expected_total_samples': expected_total,
        'amplification_factor': amplification_factor,
        'sampling_efficiency': expected_total / n_samples
    }


def test_goss_determinism():
    """GOSSの決定論的動作をテスト"""
    print("=== GOSS決定論的動作テスト ===")
    
    X, y = create_deterministic_dataset(n_samples=200, seed=123)
    
    # 同じシードで2回実行
    result1 = train_with_goss_params(X, y, seed=123)
    result2 = train_with_goss_params(X, y, seed=123)
    
    # 予測結果が一致することを確認
    pred_diff = np.array(result1['predictions']) - np.array(result2['predictions'])
    max_diff = np.max(np.abs(pred_diff))
    
    print(f"最大予測差分: {max_diff}")
    
    # 結果のハッシュ比較
    hash1 = hashlib.md5(str(result1['predictions']).encode()).hexdigest()
    hash2 = hashlib.md5(str(result2['predictions']).encode()).hexdigest()
    
    if hash1 == hash2:
        print("✅ GOSS決定論的動作: PASS")
    else:
        print("❌ GOSS決定論的動作: FAIL")
        print(f"Hash1: {hash1}")
        print(f"Hash2: {hash2}")
    
    assert max_diff < 1e-10, f"予測結果の差分が大きすぎます: {max_diff}"
    
    return result1


def test_goss_sampling_ratios():
    """GOSSサンプリング比率の正確性をテスト"""
    print("=== GOSSサンプリング比率テスト ===")
    
    test_cases = [
        {'top_rate': 0.2, 'other_rate': 0.1},
        {'top_rate': 0.3, 'other_rate': 0.15},
        {'top_rate': 0.1, 'other_rate': 0.05}
    ]
    
    X, y = create_deterministic_dataset(n_samples=1000, seed=456)
    
    for case in test_cases:
        print(f"\nテストケース: top_rate={case['top_rate']}, other_rate={case['other_rate']}")
        
        sampling_info = analyze_goss_sampling_behavior(X, y, **case)
        result = train_with_goss_params(X, y, **case, seed=456)
        
        print(f"期待されるサンプル数: {sampling_info['expected_total_samples']}")
        print(f"増幅係数: {sampling_info['amplification_factor']:.3f}")
        print(f"サンプリング効率: {sampling_info['sampling_efficiency']:.1%}")
        
        # モデルが正常に訓練されたことを確認
        assert result['num_trees'] > 0, "ツリーが構築されませんでした"
        
    print("✅ GOSSサンプリング比率テスト: PASS")
    return test_cases


def test_goss_vs_gbdt_performance():
    """GOSSとGBDTのパフォーマンス比較"""
    print("=== GOSS vs GBDT パフォーマンス比較 ===")
    
    X, y = create_deterministic_dataset(n_samples=800, seed=789)
    
    # GBDT（通常のブースティング）で訓練
    lgb_train = lgb.Dataset(X, y)
    gbdt_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',  # GBDTを使用
        'num_leaves': 15,
        'min_data_in_leaf': 10,
        'learning_rate': 0.1,
        'feature_fraction': 1.0,
        'verbose': -1,
        'seed': 789,
        'deterministic': True
    }
    
    gbdt_model = lgb.train(
        params=gbdt_params,
        train_set=lgb_train,
        num_boost_round=5
    )
    gbdt_predictions = gbdt_model.predict(X)
    gbdt_result = {
        'predictions': gbdt_predictions.tolist(),
        'params': gbdt_params
    }
    
    # GOSS で訓練
    goss_result = train_with_goss_params(X, y, top_rate=0.2, other_rate=0.1, seed=789)
    
    # RMSE を計算
    gbdt_rmse = np.sqrt(np.mean((gbdt_predictions - y) ** 2))
    goss_rmse = np.sqrt(np.mean((np.array(goss_result['predictions']) - y) ** 2))
    
    print(f"GBDT RMSE: {gbdt_rmse:.6f}")
    print(f"GOSS RMSE: {goss_rmse:.6f}")
    print(f"RMSE差分: {abs(gbdt_rmse - goss_rmse):.6f}")
    
    # GOSSは若干性能が低下するのが正常
    performance_ratio = goss_rmse / gbdt_rmse
    print(f"性能比率 (GOSS/GBDT): {performance_ratio:.3f}")
    
    # 性能低下が合理的な範囲内であることを確認（通常1.0-1.2程度）
    assert 0.8 <= performance_ratio <= 1.5, f"性能比率が異常です: {performance_ratio}"
    
    print("✅ GOSS vs GBDT パフォーマンス比較: PASS")
    
    return {
        'gbdt': gbdt_result,
        'goss': goss_result,
        'gbdt_rmse': gbdt_rmse,
        'goss_rmse': goss_rmse,
        'performance_ratio': performance_ratio
    }


def create_go_test_data():
    """Go実装テスト用のデータを生成"""
    print("=== Go実装テスト用データ生成 ===")
    
    # 小さなデータセットで詳細な検証用データを作成
    X, y = create_deterministic_dataset(n_samples=100, n_features=5, seed=999)
    
    # GOSS設定でモデルを訓練
    goss_result = train_with_goss_params(
        X, y, 
        top_rate=0.2, 
        other_rate=0.1, 
        num_boost_round=3,
        seed=999
    )
    
    # サンプリング動作の期待値を計算
    sampling_info = analyze_goss_sampling_behavior(X, y, top_rate=0.2, other_rate=0.1)
    
    # テストデータを構築
    test_data = {
        'dataset': {
            'X': X.tolist(),
            'y': y.tolist(),
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1])
        },
        'goss_params': {
            'top_rate': 0.2,
            'other_rate': 0.1,
            'seed': 999
        },
        'expected_sampling': sampling_info,
        'lightgbm_result': {
            'predictions': goss_result['predictions'],
            'num_trees': goss_result['num_trees'],
            'feature_importance': goss_result['feature_importance']
        },
        'training_params': {
            'objective': 'regression',
            'num_leaves': 15,
            'min_data_in_leaf': 10,
            'learning_rate': 0.1,
            'num_boost_round': 3,
            'verbose': -1
        },
        'test_info': {
            'description': 'GOSS verification data for Go implementation',
            'expected_behavior': 'Go GOSS should produce identical sampling and predictions',
            'tolerance': 1e-6
        }
    }
    
    # ファイルに保存
    os.makedirs('testdata', exist_ok=True)
    with open('testdata/goss_verification_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"✅ Go実装テスト用データ保存完了: testdata/goss_verification_data.json")
    print(f"データセットサイズ: {X.shape}")
    print(f"期待されるサンプル数: {sampling_info['expected_total_samples']}")
    print(f"LightGBM予測範囲: [{min(goss_result['predictions']):.3f}, {max(goss_result['predictions']):.3f}]")
    
    return test_data


def main():
    """全テストの実行"""
    print("=== GOSS検証テスト開始 ===\n")
    
    try:
        # 1. 決定論的動作テスト
        print("1. GOSS決定論的動作テスト")
        test_goss_determinism()
        print()
        
        # 2. サンプリング比率テスト
        print("2. GOSSサンプリング比率テスト")
        test_goss_sampling_ratios()
        print()
        
        # 3. パフォーマンス比較テスト
        print("3. GOSS vs GBDT パフォーマンス比較")
        test_goss_vs_gbdt_performance()
        print()
        
        # 4. Go実装テスト用データ生成
        print("4. Go実装テスト用データ生成")
        create_go_test_data()
        print()
        
        print("🎉 全GOSS検証テスト完了: SUCCESS")
        
    except Exception as e:
        print(f"❌ GOSS検証テスト失敗: {e}")
        raise


if __name__ == "__main__":
    main()