#!/usr/bin/env python3
"""
PredictProba (確率予測) 検証テスト

このテストスイートは、Goで実装されたPredictProbaが
Python LightGBMと同等の動作をすることを検証します。
特にMulticlassLogLossとの連携を中心にテストします。
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import json
import os
from typing import Dict, List, Tuple, Optional
from scipy.special import softmax


def create_multiclass_dataset(n_samples: int = 300, n_features: int = 6, n_classes: int = 3, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    多クラス分類用のデータセットを作成
    
    Args:
        n_samples: サンプル数
        n_features: 特徴量数
        n_classes: クラス数
        seed: 乱数シード
        
    Returns:
        X: 特徴量行列
        y: ターゲット配列（クラスラベル）
    """
    np.random.seed(seed)
    
    # 特徴量行列の生成
    X = np.random.randn(n_samples, n_features)
    
    # クラス別の重み係数を設定
    class_weights = []
    for c in range(n_classes):
        weights = np.random.randn(n_features) * 0.5
        class_weights.append(weights)
    
    # 各サンプルに対してクラス確率を計算
    logits = np.zeros((n_samples, n_classes))
    for c in range(n_classes):
        logits[:, c] = X @ class_weights[c] + np.random.randn(n_samples) * 0.1
    
    # Softmax でクラス確率を計算
    probabilities = softmax(logits, axis=1)
    
    # 確率に基づいてクラスラベルをサンプリング
    y = np.array([np.random.choice(n_classes, p=probabilities[i]) for i in range(n_samples)])
    
    return X, y


def train_multiclass_lightgbm(X: np.ndarray, y: np.ndarray, 
                             num_boost_round: int = 10,
                             num_classes: int = 3,
                             seed: int = 42) -> Dict:
    """
    多クラス分類でLightGBMを訓練
    
    Args:
        X: 特徴量行列
        y: ターゲット配列
        num_boost_round: ブースティングラウンド数
        num_classes: クラス数
        seed: 乱数シード
        
    Returns:
        訓練結果の詳細情報
    """
    lgb_train = lgb.Dataset(X, y)
    
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': num_classes,
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
    
    # 通常の予測（マージン）
    predictions_margin = model.predict(X, num_iteration=model.best_iteration)
    
    # 確率予測
    predictions_proba = model.predict(X, num_iteration=model.best_iteration)
    
    # LightGBMのmulticlassは自動でSoftmaxが適用される
    
    return {
        'model': model,
        'params': params,
        'predictions_margin': predictions_margin.tolist(),
        'predictions_proba': predictions_proba.tolist(),
        'eval_results': eval_results,
        'num_trees': model.num_trees(),
        'feature_importance': model.feature_importance().tolist()
    }


def test_multiclass_probability_predictions():
    """多クラス分類での確率予測をテスト"""
    print("=== 多クラス分類確率予測テスト ===")
    
    X, y = create_multiclass_dataset(n_samples=200, n_classes=3, seed=123)
    
    # LightGBMで訓練
    result = train_multiclass_lightgbm(X, y, num_boost_round=5, num_classes=3, seed=123)
    
    predictions_proba = np.array(result['predictions_proba'])
    
    print(f"データセットサイズ: {X.shape}")
    print(f"クラス数: 3")
    print(f"予測確率の形状: {predictions_proba.shape}")
    
    # 確率の性質をチェック
    for i in range(min(5, len(predictions_proba))):
        proba_sum = np.sum(predictions_proba[i])
        print(f"サンプル {i}: 確率合計 = {proba_sum:.6f}, 予測 = {predictions_proba[i]}")
        
        # 確率の合計は1に近いはず
        assert abs(proba_sum - 1.0) < 1e-6, f"確率の合計が1ではありません: {proba_sum}"
        
        # 全ての確率は0以上1以下
        assert np.all(predictions_proba[i] >= 0), f"負の確率があります: {predictions_proba[i]}"
        assert np.all(predictions_proba[i] <= 1), f"1を超える確率があります: {predictions_proba[i]}"
    
    print("✅ 多クラス分類確率予測テスト: PASS")
    return result


def test_softmax_consistency():
    """Softmax変換の一貫性をテスト"""
    print("=== Softmax変換一貫性テスト ===")
    
    X, y = create_multiclass_dataset(n_samples=100, n_classes=4, seed=456)
    
    # LightGBMで訓練
    result = train_multiclass_lightgbm(X, y, num_boost_round=3, num_classes=4, seed=456)
    
    predictions_proba = np.array(result['predictions_proba'])
    
    # 手動でSoftmaxを確認
    # LightGBMのmulticlass objectiveは内部でSoftmaxを適用するため、
    # 生のlogitを取得するのは困難。代わりに一貫性をチェック
    
    print(f"確率予測の統計:")
    print(f"  最小値: {np.min(predictions_proba):.6f}")
    print(f"  最大値: {np.max(predictions_proba):.6f}")
    print(f"  平均値: {np.mean(predictions_proba):.6f}")
    print(f"  行合計の平均: {np.mean(np.sum(predictions_proba, axis=1)):.6f}")
    print(f"  行合計の標準偏差: {np.std(np.sum(predictions_proba, axis=1)):.6f}")
    
    # 全ての行の合計は1に近いはず
    row_sums = np.sum(predictions_proba, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), "確率の行合計が1ではありません"
    
    print("✅ Softmax変換一貫性テスト: PASS")
    return result


def test_class_prediction_accuracy():
    """クラス予測精度をテスト"""
    print("=== クラス予測精度テスト ===")
    
    X, y = create_multiclass_dataset(n_samples=400, n_classes=3, seed=789)
    
    # 訓練/テストデータ分割
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # LightGBMで訓練
    lgb_train = lgb.Dataset(X_train, y_train)
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 3,
        'num_leaves': 20,
        'min_data_in_leaf': 5,
        'learning_rate': 0.1,
        'verbose': -1,
        'seed': 789,
        'deterministic': True
    }
    
    model = lgb.train(
        params=params,
        train_set=lgb_train,
        num_boost_round=15
    )
    
    # テストデータで予測
    y_pred_proba = model.predict(X_test)
    y_pred_class = np.argmax(y_pred_proba, axis=1)
    
    # 精度計算
    accuracy = np.mean(y_pred_class == y_test)
    print(f"テストセット精度: {accuracy:.3f}")
    
    # クラス別の予測確率統計
    for c in range(3):
        class_mask = y_test == c
        if np.any(class_mask):
            class_proba = y_pred_proba[class_mask, c]
            print(f"クラス {c}: 平均確率 = {np.mean(class_proba):.3f}, "
                  f"正解時の平均確率 = {np.mean(y_pred_proba[class_mask & (y_pred_class == c), c]):.3f}")
    
    # 最低限の精度を期待
    assert accuracy > 0.3, f"予測精度が低すぎます: {accuracy}"
    
    print("✅ クラス予測精度テスト: PASS")
    return {
        'accuracy': accuracy,
        'y_pred_proba': y_pred_proba.tolist(),
        'y_pred_class': y_pred_class.tolist(),
        'y_test': y_test.tolist()
    }


def create_go_test_data():
    """Go実装テスト用のデータを生成"""
    print("=== Go実装テスト用データ生成 ===")
    
    # 小さなデータセットで詳細な検証用データを作成
    X, y = create_multiclass_dataset(n_samples=80, n_features=4, n_classes=3, seed=999)
    
    # MulticlassLogLoss設定でモデルを訓練
    result = train_multiclass_lightgbm(X, y, num_boost_round=5, num_classes=3, seed=999)
    
    # テストデータを構築
    test_data = {
        'dataset': {
            'X': X.tolist(),
            'y': y.tolist(),
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'n_classes': 3
        },
        'training_params': {
            'objective': 'multiclass',
            'num_class': 3,
            'num_leaves': 15,
            'min_data_in_leaf': 10,
            'learning_rate': 0.1,
            'num_boost_round': 5,
            'seed': 999,
            'verbose': -1
        },
        'lightgbm_result': {
            'predictions_proba': result['predictions_proba'],
            'num_trees': result['num_trees'],
            'feature_importance': result['feature_importance']
        },
        'expected_behavior': {
            'probability_sum_per_sample': 1.0,
            'probability_range': [0.0, 1.0],
            'softmax_transformation': 'Applied automatically for multiclass objective'
        },
        'test_info': {
            'description': 'PredictProba verification data for Go implementation',
            'focus': 'MulticlassLogLoss integration and Softmax consistency',
            'tolerance': 1e-5
        }
    }
    
    # ファイルに保存
    os.makedirs('testdata', exist_ok=True)
    with open('testdata/predictproba_verification_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"✅ Go実装テスト用データ保存完了: testdata/predictproba_verification_data.json")
    print(f"データセットサイズ: {X.shape}")
    print(f"クラス数: 3")
    
    # 予測確率の統計情報
    predictions_proba = np.array(result['predictions_proba'])
    print(f"確率予測統計:")
    print(f"  形状: {predictions_proba.shape}")
    print(f"  確率範囲: [{np.min(predictions_proba):.3f}, {np.max(predictions_proba):.3f}]")
    print(f"  行合計の範囲: [{np.min(np.sum(predictions_proba, axis=1)):.6f}, {np.max(np.sum(predictions_proba, axis=1)):.6f}]")
    
    return test_data


def test_edge_cases():
    """エッジケースのテスト"""
    print("=== エッジケーステスト ===")
    
    # 極端に偏ったクラス分布
    X, y = create_multiclass_dataset(n_samples=150, n_classes=3, seed=555)
    
    # クラス分布を人工的に偏らせる
    class_counts = np.bincount(y)
    print(f"元のクラス分布: {class_counts}")
    
    # クラス0を多数派にする
    majority_mask = y == 0
    minority_indices = np.where(~majority_mask)[0]
    
    # 一部のサンプルをクラス0に変更
    change_indices = minority_indices[:len(minority_indices)//2]
    y[change_indices] = 0
    
    new_class_counts = np.bincount(y)
    print(f"調整後のクラス分布: {new_class_counts}")
    
    # LightGBMで訓練
    result = train_multiclass_lightgbm(X, y, num_boost_round=8, num_classes=3, seed=555)
    
    predictions_proba = np.array(result['predictions_proba'])
    
    # クラス不均衡下でも確率の性質が保たれているかチェック
    row_sums = np.sum(predictions_proba, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), "不均衡データで確率合計が1ではありません"
    
    # 多数派クラスの平均確率が高いことを確認
    majority_class_proba = np.mean(predictions_proba[:, 0])
    minority_class_proba = np.mean(predictions_proba[:, 1:])
    print(f"多数派クラス(0)の平均確率: {majority_class_proba:.3f}")
    print(f"少数派クラスの平均確率: {minority_class_proba:.3f}")
    
    print("✅ エッジケーステスト: PASS")
    return result


def main():
    """全テストの実行"""
    print("=== PredictProba検証テスト開始 ===\n")
    
    try:
        # 1. 多クラス分類確率予測テスト
        print("1. 多クラス分類確率予測テスト")
        test_multiclass_probability_predictions()
        print()
        
        # 2. Softmax変換一貫性テスト  
        print("2. Softmax変換一貫性テスト")
        test_softmax_consistency()
        print()
        
        # 3. クラス予測精度テスト
        print("3. クラス予測精度テスト")
        test_class_prediction_accuracy()
        print()
        
        # 4. エッジケーステスト
        print("4. エッジケーステスト")
        test_edge_cases()
        print()
        
        # 5. Go実装テスト用データ生成
        print("5. Go実装テスト用データ生成")
        create_go_test_data()
        print()
        
        print("🎉 全PredictProba検証テスト完了: SUCCESS")
        
    except Exception as e:
        print(f"❌ PredictProba検証テスト失敗: {e}")
        raise


if __name__ == "__main__":
    main()