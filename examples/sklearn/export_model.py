#!/usr/bin/env python3
"""
scikit-learnモデルをSciGo互換のJSON形式でエクスポートするスクリプト

使用例:
    python export_model.py

このスクリプトは：
1. scikit-learnで線形回帰モデルを学習
2. SciGo互換のJSON形式でエクスポート
3. エクスポートしたモデルの検証
"""

import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import __version__ as sklearn_version


def export_linear_regression(model, filename):
    """
    LinearRegressionモデルをSciGo互換のJSON形式でエクスポート
    
    Args:
        model: 学習済みのLinearRegressionモデル
        filename: 出力ファイル名
    """
    # モデルのパラメータを取得
    coefficients = model.coef_.tolist()
    intercept = float(model.intercept_)
    n_features = len(coefficients)
    
    # SciGo互換のJSON構造を作成
    model_json = {
        "model_spec": {
            "name": "LinearRegression",
            "format_version": "1.0",
            "sklearn_version": sklearn_version
        },
        "params": {
            "coefficients": coefficients,
            "intercept": intercept,
            "n_features": n_features
        }
    }
    
    # JSONファイルに保存
    with open(filename, 'w') as f:
        json.dump(model_json, f, indent=2)
    
    print(f"Model exported to {filename}")
    return model_json


def verify_export(original_model, json_file, X_test):
    """
    エクスポートしたモデルの検証
    
    Args:
        original_model: 元のscikit-learnモデル
        json_file: エクスポートしたJSONファイル
        X_test: テストデータ
    """
    # JSONファイルを読み込み
    with open(json_file, 'r') as f:
        loaded = json.load(f)
    
    # パラメータを取得
    coefficients = np.array(loaded['params']['coefficients'])
    intercept = loaded['params']['intercept']
    
    # 手動で予測を計算
    manual_predictions = X_test @ coefficients + intercept
    
    # scikit-learnモデルの予測
    sklearn_predictions = original_model.predict(X_test)
    
    # 予測が一致することを確認
    max_diff = np.max(np.abs(manual_predictions - sklearn_predictions))
    print(f"Maximum difference between predictions: {max_diff}")
    
    if max_diff < 1e-10:
        print("✓ Export verified successfully!")
    else:
        print("✗ Export verification failed!")
    
    return manual_predictions, sklearn_predictions


def main():
    # サンプルデータを生成
    np.random.seed(42)
    n_samples, n_features = 100, 3
    
    # データ生成（y = 2*x1 + 3*x2 - 1*x3 + 5 + noise）
    X = np.random.randn(n_samples, n_features)
    true_coefficients = np.array([2.0, 3.0, -1.0])
    true_intercept = 5.0
    noise = np.random.randn(n_samples) * 0.1
    y = X @ true_coefficients + true_intercept + noise
    
    # モデルの学習
    print("Training LinearRegression model...")
    model = LinearRegression()
    model.fit(X, y)
    
    # 学習したパラメータを表示
    print(f"\nLearned coefficients: {model.coef_}")
    print(f"Learned intercept: {model.intercept_}")
    
    # モデルをエクスポート
    print("\nExporting model...")
    json_file = "linear_regression_model.json"
    export_linear_regression(model, json_file)
    
    # エクスポートしたモデルの検証
    print("\nVerifying exported model...")
    X_test = np.random.randn(5, n_features)
    verify_export(model, json_file, X_test)
    
    # テストデータと予測結果を保存（Go側でのテスト用）
    test_data = {
        "X_test": X_test.tolist(),
        "y_pred": model.predict(X_test).tolist()
    }
    
    with open("test_data.json", 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print("\nTest data saved to test_data.json")
    print("\nExample usage in Go:")
    print("  model, _ := model.LoadSKLearnModelFromFile(\"linear_regression_model.json\")")
    print("  params, _ := model.LoadLinearRegressionParams(model)")
    print("  // Use params to initialize LinearRegression in Go")


if __name__ == "__main__":
    main()