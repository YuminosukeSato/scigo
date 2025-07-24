# GoML エラーハンドリングガイドライン

## 概要

GoMLプロジェクトでは、エラーハンドリングを以下の原則に基づいて実装しています：

1. **レイヤー別の責務分離**
2. **構造化されたエラー情報**
3. **適切なログ出力**

## レイヤー別の責務

### 1. pkg/errors パッケージ（エラーハンドリング統合層）

**責務**: エラーの定義、生成、操作の統一窓口

- エラー型の定義（`DimensionError`, `NotFittedError`など）
- エラーの生成（`New*`関数）
- スタックトレースの付与
- エラーの判定（`Is`, `As`）
- エラーのラップ（`Wrap`, `Wrapf`）
- エラー詳細情報の抽出（`GetDetails`, `GetStackTrace`）

```go
import "github.com/YuminosukeSato/GoML/pkg/errors"

// エラーの生成
return errors.NewDimensionError("Predict", []int{10, 3}, []int{10, 2})
```

### 2. core パッケージ（機械学習ロジック層）

**責務**: 機械学習ロジックの実装

- モデルの実装
- pkg/errorsを使用してエラーを生成
- **ログ出力は行わない**

```go
import "github.com/YuminosukeSato/GoML/pkg/errors"

// エラーの判定
if errors.Is(err, errors.ErrNotFitted) {
    // 処理
}

// エラーのラップ
return errors.Wrap(err, "failed to process data")

// エラー型のキャスト
var dimErr *errors.DimensionError
if errors.As(err, &dimErr) {
    // dimErrを使った処理
}
```

### 3. アプリケーション層（main, cmd, examples）

**責務**: エラーの最終処理とログ出力

- エラーハンドリングの判断
- slogを使用した構造化ログ出力
- ユーザーへのエラー表示

```go
import (
    "log/slog"
    "github.com/YuminosukeSato/GoML/pkg/errors"
    "github.com/YuminosukeSato/GoML/pkg/log"
)

if err != nil {
    // 構造化ログ出力
    slog.Error("Failed to predict", 
        log.ErrAttr(err),
        slog.String("model", "LinearRegression"),
        slog.Int("features", 10),
    )
    
    // エラー詳細の取得
    details := errors.GetAllDetails(err)
    slog.Debug("Error details", slog.Any("details", details))
}
```

## エラー分類と処理方針

### 1. ユーザー起因のエラー

- **例**: `DimensionError`, `ValueError`
- **ログレベル**: `slog.Warn` または `slog.Info`
- **処理**: わかりやすいエラーメッセージをユーザーに返す

### 2. システムエラー

- **例**: `NotFittedError`, `ConvergenceError`
- **ログレベル**: `slog.Error`
- **処理**: スタックトレース付きでログ出力、開発者向けの詳細情報を記録

### 3. 致命的エラー

- **ログレベル**: `slog.Error`
- **処理**: ログ出力後、必要に応じて`os.Exit(1)`

## ベストプラクティス

### 1. importの推奨方法

```go
import "github.com/YuminosukeSato/GoML/pkg/errors"

// エラー型、生成関数、操作関数すべてがpkg/errorsから使用可能
```

### 2. エラーチェーンの構築

```go
// 下位層でのエラー
err := doSomething()
if err != nil {
    return errors.Wrap(err, "failed to do something")
}

// 上位層でさらにコンテキストを追加
err = processData()
if err != nil {
    return errors.Wrapf(err, "failed to process data for user %d", userID)
}
```

### 3. エラー詳細のログ出力

```go
// 基本的なエラーログ
slog.Error("Operation failed", log.ErrAttr(err))

// 詳細情報付きのエラーログ
details := errors.GetAllDetails(err)
slog.Error("Operation failed",
    log.ErrAttr(err),
    slog.Any("error_details", details),
    slog.String("operation", "model_training"),
)
```

## 禁止事項

1. **cockroachdb/errorsを直接使用しない（pkg/errorsを使用）**
2. **本番環境でのログにセンシティブ情報を含めない**
3. **エラーを握りつぶさない（必ずログまたは上位層に伝播）**
4. **core層でログ出力しない**

## 移行ガイド

既存のコードを新しいエラーハンドリング方式に移行する場合：

1. `import gomlErrors "github.com/YuminosukeSato/GoML/core/errors"` → `import "github.com/YuminosukeSato/GoML/pkg/errors"`
2. `gomlErrors.NewDimensionError` → `errors.NewDimensionError`
3. `gomlErrors.ErrNotFitted` → `errors.ErrNotFitted`
4. エイリアスを削除して、すべて`errors`で統一

## まとめ

- **pkg/errors**: エラーの定義、生成、操作のすべてを提供
- **coreパッケージ**: pkg/errorsを使用してエラーを生成
- **アプリケーション層**: slogでのログ出力

この構造により、エラーハンドリングがpkg/errorsに一元化され、保守性と使いやすさが向上します。