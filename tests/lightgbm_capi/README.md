# LightGBM C API検証環境

このディレクトリは、純Go実装のLightGBMとPython/C API版のLightGBMの結果を比較検証するための環境です。

## 目的

- 純Go実装の正確性を検証
- Python LightGBMとの互換性を確認
- 数値精度の差異を詳細に分析

## ディレクトリ構造

```
tests/lightgbm_capi/
├── capi_wrapper.go          # CGOでLightGBM C APIをラップ
├── comparison_test.go       # 純Go vs C API vs Python比較テスト
├── python_baseline/         # Python基準値生成
│   ├── train_models.py      # モデル学習スクリプト
│   ├── generate_baseline.py # 基準値生成
│   └── requirements.txt     # 依存パッケージ
├── testdata/               # 共通テストデータ
│   ├── regression/         # 回帰用データ
│   ├── binary/            # 二値分類用データ
│   └── multiclass/        # 多クラス分類用データ
└── README.md              # このファイル
```

## セットアップ

### 1. LightGBM C++ライブラリのインストール

macOS:
```bash
brew install lightgbm
```

Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
mkdir build && cd build
cmake ..
make -j4
sudo make install
```

### 2. Python環境のセットアップ

```bash
cd python_baseline
pip install -r requirements.txt
```

## 使用方法

### 1. Python基準値の生成

```bash
cd python_baseline
python train_models.py      # モデルを学習
python generate_baseline.py  # 基準値を生成
```

### 2. 比較テストの実行

```bash
cd ..
go test -v ./...
```

## 検証項目

### 予測の一致
- 同じモデルファイルから同じ予測値が得られるか
- 葉ノードインデックスが一致するか
- 特徴量重要度が一致するか

### 学習の一致
- deterministicモードで同じ木構造が得られるか
- 勾配・ヘシアンの計算が一致するか
- 分割点の選択が一致するか

## 数値精度の基準

- 予測値: 相対誤差 < 1e-9
- 勾配・ヘシアン: 絶対誤差 < 1e-10
- 葉ノード値: 相対誤差 < 1e-8

## 注意事項

- CGOを使用するため、ビルド時にLightGBMライブラリが必要
- deterministicモードでもバージョンやコンパイラによって微小な差が生じる可能性がある
- 完全一致を目指すが、実用上は許容誤差内での一致を目標とする