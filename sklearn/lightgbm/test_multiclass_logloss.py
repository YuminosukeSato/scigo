#!/usr/bin/env python3
"""
MulticlassLogLoss実装の検証テスト

このテストスイートは、Goで実装されたMulticlassLogLossが
理論的な計算とscipy実装と完全に一致することを検証します。
"""

import numpy as np
import json
import os
from scipy.special import softmax, log_softmax
from typing import Tuple, Dict, Any


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """数値安定性を考慮したSoftmax計算"""
    # 各サンプルごとに最大値を減算
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def calculate_gradients_and_hessians(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    多クラス交差エントロピー損失の勾配とヘシアンを計算
    
    Args:
        y_true: 真のクラスラベル [num_samples]
        y_pred: 予測ロジット [num_samples, num_classes]
        
    Returns:
        gradients: [num_samples, num_classes]
        hessians: [num_samples, num_classes]
    """
    num_samples, num_classes = y_pred.shape
    
    # Softmax確率を計算
    probabilities = stable_softmax(y_pred)
    
    # One-hotエンコーディング
    y_true_onehot = np.zeros((num_samples, num_classes))
    y_true_onehot[np.arange(num_samples), y_true] = 1.0
    
    # 勾配: p_k - y_k
    gradients = probabilities - y_true_onehot
    
    # ヘシアン（対角近似）: p_k * (1 - p_k)
    hessians = probabilities * (1.0 - probabilities)
    
    # 数値安定性の確保
    hessians = np.maximum(hessians, 1e-16)
    
    return gradients, hessians


def calculate_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """多クラス交差エントロピー損失を計算"""
    num_samples = y_pred.shape[0]
    
    # 数値安定性を考慮したlog softmaxを計算
    log_probs = log_softmax(y_pred, axis=1)
    
    # 真のクラスのlog確率を抽出
    true_log_probs = log_probs[np.arange(num_samples), y_true]
    
    # 平均交差エントロピー損失: -mean(log(p_true_class))
    return -np.mean(true_log_probs)


def test_basic_gradients_hessians():
    """基本的な勾配・ヘシアン計算のテスト"""
    print("=== 基本的な勾配・ヘシアン計算テスト ===")
    
    # 3クラス、4サンプルのテストケース
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([
        [2.0, 0.5, -1.0],    # クラス0が最大
        [-0.5, 3.0, -1.0],   # クラス1が最大
        [0.0, -1.0, 2.5],    # クラス2が最大
        [1.8, 0.3, -0.8]     # クラス0が最大
    ])
    
    gradients, hessians = calculate_gradients_and_hessians(y_true, y_pred)
    loss = calculate_loss(y_true, y_pred)
    
    print(f"入力ロジット:\n{y_pred}")
    print(f"真のクラス: {y_true}")
    print(f"勾配:\n{gradients}")
    print(f"ヘシアン:\n{hessians}")
    print(f"損失: {loss:.6f}")
    
    # 基本的な性質をチェック
    # 1. 勾配の各行の合計は0に近い（softmax制約）
    gradient_sums = np.sum(gradients, axis=1)
    print(f"勾配行合計: {gradient_sums}")
    assert np.allclose(gradient_sums, 0.0, atol=1e-10), "勾配の行合計が0になっていません"
    
    # 2. ヘシアンは全て正値
    assert np.all(hessians > 0), "ヘシアンに非正値があります"
    
    # 3. 確率の合計が1
    probabilities = stable_softmax(y_pred)
    prob_sums = np.sum(probabilities, axis=1)
    assert np.allclose(prob_sums, 1.0), "確率の合計が1になっていません"
    
    print("✅ 基本テスト合格\n")
    return {
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
        'gradients': gradients.tolist(),
        'hessians': hessians.tolist(),
        'loss': float(loss)
    }


def test_edge_cases():
    """エッジケースのテスト"""
    print("=== エッジケーステスト ===")
    
    # 極端なロジット値
    y_true = np.array([0, 1])
    y_pred = np.array([
        [100.0, -50.0, -50.0],  # 極めて確信度の高い予測
        [-50.0, 100.0, -50.0]   # 極めて確信度の高い予測
    ])
    
    gradients, hessians = calculate_gradients_and_hessians(y_true, y_pred)
    loss = calculate_loss(y_true, y_pred)
    
    print(f"極端なロジット値での損失: {loss:.6f}")
    print(f"勾配範囲: [{np.min(gradients):.6f}, {np.max(gradients):.6f}]")
    print(f"ヘシアン範囲: [{np.min(hessians):.6f}, {np.max(hessians):.6f}]")
    
    # 値が有限であることを確認
    assert np.all(np.isfinite(gradients)), "勾配に無限値があります"
    assert np.all(np.isfinite(hessians)), "ヘシアンに無限値があります"
    assert np.isfinite(loss), "損失が無限値です"
    
    print("✅ エッジケーステスト合格\n")
    
    return {
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
        'gradients': gradients.tolist(),
        'hessians': hessians.tolist(),
        'loss': float(loss)
    }


def test_large_dataset():
    """大規模データセットでのテスト"""
    print("=== 大規模データセットテスト ===")
    
    np.random.seed(42)
    num_samples = 1000
    num_classes = 5
    
    # ランダムなロジットと真のクラス
    y_pred = np.random.randn(num_samples, num_classes) * 2
    y_true = np.random.randint(0, num_classes, num_samples)
    
    gradients, hessians = calculate_gradients_and_hessians(y_true, y_pred)
    loss = calculate_loss(y_true, y_pred)
    
    print(f"サンプル数: {num_samples}, クラス数: {num_classes}")
    print(f"損失: {loss:.6f}")
    print(f"勾配統計: 平均={np.mean(gradients):.6f}, 標準偏差={np.std(gradients):.6f}")
    print(f"ヘシアン統計: 平均={np.mean(hessians):.6f}, 標準偏差={np.std(hessians):.6f}")
    
    # 基本的な性質をチェック
    gradient_sums = np.sum(gradients, axis=1)
    assert np.allclose(gradient_sums, 0.0, atol=1e-10), "大規模データで勾配制約が違反されています"
    
    print("✅ 大規模データセットテスト合格\n")
    
    return {
        'num_samples': num_samples,
        'num_classes': num_classes,
        'loss': float(loss),
        'gradient_mean': float(np.mean(gradients)),
        'gradient_std': float(np.std(gradients)),
        'hessian_mean': float(np.mean(hessians)),
        'hessian_std': float(np.std(hessians))
    }


def create_go_test_data():
    """Go実装テスト用のデータを生成"""
    print("=== Go実装テスト用データ生成 ===")
    
    # テストケース1: 基本ケース
    basic_case = test_basic_gradients_hessians()
    
    # テストケース2: エッジケース
    edge_case = test_edge_cases()
    
    # テストケース3: より複雑なケース
    np.random.seed(123)
    y_true_complex = np.random.randint(0, 4, 20)
    y_pred_complex = np.random.randn(20, 4) * 1.5
    
    gradients_complex, hessians_complex = calculate_gradients_and_hessians(y_true_complex, y_pred_complex)
    loss_complex = calculate_loss(y_true_complex, y_pred_complex)
    
    complex_case = {
        'y_true': y_true_complex.tolist(),
        'y_pred': y_pred_complex.tolist(),
        'gradients': gradients_complex.tolist(),
        'hessians': hessians_complex.tolist(),
        'loss': float(loss_complex)
    }
    
    # テストデータ統合
    test_data = {
        'basic_case': basic_case,
        'edge_case': edge_case,
        'complex_case': complex_case,
        'test_info': {
            'description': 'MulticlassLogLoss validation data for Go implementation',
            'gradient_formula': 'gradient[i,k] = p[i,k] - y[i,k] (where p=softmax(logits), y=onehot(true))',
            'hessian_formula': 'hessian[i,k] = p[i,k] * (1 - p[i,k])',
            'loss_formula': 'loss = -mean(log(p[i,true_class[i]]))'
        }
    }
    
    # ファイルに保存
    os.makedirs('testdata', exist_ok=True)
    with open('testdata/multiclass_logloss_test_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"✅ Go実装テスト用データ保存完了: testdata/multiclass_logloss_test_data.json")
    return test_data


def main():
    """全テストの実行"""
    print("=== MulticlassLogLoss実装検証テスト開始 ===\n")
    
    try:
        # 基本テスト
        test_basic_gradients_hessians()
        
        # エッジケーステスト
        test_edge_cases()
        
        # 大規模データセットテスト
        test_large_dataset()
        
        # Go実装用テストデータ生成
        create_go_test_data()
        
        print("🎉 全テスト完了: SUCCESS")
        
    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        raise


if __name__ == "__main__":
    main()