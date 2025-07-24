// Package errors はプロジェクト全体のエラーハンドリングを提供します。
// このパッケージには、エラー型定義、エラー生成関数、およびエラー操作関数が含まれています。
// cockroachdb/errorsを直接使用せず、このパッケージを通じてエラー処理を行ってください。
package errors

import (
	"fmt"
	"strings"

	"github.com/cockroachdb/errors"
)

// ============================================================================
// エラー操作関数（cockroachdb/errorsのラッパー）
// ============================================================================

// Is はエラーが特定のターゲットエラーかどうかを判定する
func Is(err, target error) bool {
	return errors.Is(err, target)
}

// As はエラーが特定の型にキャスト可能かどうかを判定する
func As(err error, target interface{}) bool {
	return errors.As(err, target)
}

// Wrap は既存のエラーをラップする
func Wrap(err error, message string) error {
	return errors.Wrap(err, message)
}

// Wrapf は既存のエラーをフォーマット文字列でラップする
func Wrapf(err error, format string, args ...interface{}) error {
	return errors.Wrapf(err, format, args...)
}

// New は新しいエラーを作成する
func New(message string) error {
	return errors.New(message)
}

// Newf は新しいフォーマット済みエラーを作成する
func Newf(format string, args ...interface{}) error {
	return errors.Newf(format, args...)
}

// WithStack は既存のエラーにスタックトレースを付与する
func WithStack(err error) error {
	return errors.WithStack(err)
}

// ============================================================================
// エラー詳細情報抽出ヘルパー
// ============================================================================

// GetStackTrace はエラーからスタックトレースを抽出する
func GetStackTrace(err error) string {
	// フォーマッターを使ってスタックトレースを含む詳細情報を取得
	formatted := fmt.Sprintf("%+v", err)
	
	// スタックトレースが含まれているかチェック
	if strings.Contains(formatted, "stack trace:") || strings.Contains(formatted, ".go:") {
		// 簡易的にフォーマット済み文字列からスタックトレース部分を抽出
		lines := strings.Split(formatted, "\n")
		var stackLines []string
		inStack := false
		
		for _, line := range lines {
			if strings.Contains(line, "stack trace:") || (strings.Contains(line, ".go:") && strings.Contains(line, "\t")) {
				inStack = true
			}
			if inStack && strings.TrimSpace(line) != "" {
				stackLines = append(stackLines, line)
			}
		}
		
		if len(stackLines) > 0 {
			return strings.Join(stackLines, "\n")
		}
	}
	
	// cockroachdb/errorsのGetSafeDetailsも試す
	safeDetails := errors.GetSafeDetails(err).SafeDetails
	if len(safeDetails) > 0 {
		return safeDetails[0]
	}
	
	return ""
}

// ErrorDetail はエラーの詳細情報を表す構造体
type ErrorDetail struct {
	Message     string                 // エラーメッセージ
	StackTrace  string                 // スタックトレース
	Details     map[string]string      // 詳細情報
	Hints       []string               // ヒント情報
	Attributes  map[string]interface{} // その他の属性
}

// GetDetails はエラーからすべての詳細情報を抽出する
func GetDetails(err error) *ErrorDetail {
	if err == nil {
		return nil
	}

	detail := &ErrorDetail{
		Message:    err.Error(),
		StackTrace: GetStackTrace(err),
		Details:    make(map[string]string),
		Hints:      []string{},
		Attributes: make(map[string]interface{}),
	}

	// cockroachdb/errorsの詳細情報を抽出
	safeDetails := errors.GetSafeDetails(err)
	if safeDetails.SafeDetails != nil {
		for i, d := range safeDetails.SafeDetails {
			if i == 0 {
				// 最初の要素は通常スタックトレース
				continue
			}
			detail.Details[fmt.Sprintf("detail_%d", i)] = d
		}
	}

	// エラーチェーンを辿って情報を収集
	var current error = err
	for current != nil {
		// Unwrapを使ってエラーチェーンを辿る
		current = errors.Unwrap(current)
	}

	return detail
}

// GetAllDetails はエラーの詳細情報をマップ形式で返す（構造化ログ用）
func GetAllDetails(err error) map[string]interface{} {
	if err == nil {
		return nil
	}

	detail := GetDetails(err)
	result := make(map[string]interface{})

	result["message"] = detail.Message
	if detail.StackTrace != "" {
		result["stacktrace"] = detail.StackTrace
	}
	if len(detail.Details) > 0 {
		result["details"] = detail.Details
	}
	if len(detail.Hints) > 0 {
		result["hints"] = detail.Hints
	}
	if len(detail.Attributes) > 0 {
		result["attributes"] = detail.Attributes
	}

	return result
}

// ============================================================================
// エラー型定義（機械学習モデル用）
// ============================================================================

// ModelError は機械学習モデルに関するエラーを表す
type ModelError struct {
	Op   string // 操作名
	Kind string // エラーの種類
	Err  error  // 元のエラー
}

// Error はエラーメッセージを返す
func (e *ModelError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("goml: %s: %s: %v", e.Op, e.Kind, e.Err)
	}
	return fmt.Sprintf("goml: %s: %s", e.Op, e.Kind)
}

// Unwrap は元のエラーを返す
func (e *ModelError) Unwrap() error {
	return e.Err
}

// DimensionError は次元の不一致エラーを表す
type DimensionError struct {
	Op       string
	Expected []int
	Got      []int
}

// Error はエラーメッセージを返す
func (e *DimensionError) Error() string {
	return fmt.Sprintf("goml: %s: dimension mismatch: expected %v, got %v", e.Op, e.Expected, e.Got)
}

// NotFittedError はモデルが未学習の状態で予測しようとした場合のエラー
type NotFittedError struct {
	ModelName string
}

// Error はエラーメッセージを返す
func (e *NotFittedError) Error() string {
	return fmt.Sprintf("goml: %s is not fitted yet. Call Fit before Predict", e.ModelName)
}

// ValueError は値に関するエラーを表す
type ValueError struct {
	Op      string
	Param   string
	Value   interface{}
	Message string
}

// Error はエラーメッセージを返す
func (e *ValueError) Error() string {
	if e.Message != "" {
		return fmt.Sprintf("goml: %s: invalid value for %s: %v (%s)", e.Op, e.Param, e.Value, e.Message)
	}
	return fmt.Sprintf("goml: %s: invalid value for %s: %v", e.Op, e.Param, e.Value)
}

// ConvergenceError は収束しなかった場合のエラー
type ConvergenceError struct {
	ModelName  string
	Iterations int
	Message    string
}

// Error はエラーメッセージを返す
func (e *ConvergenceError) Error() string {
	if e.Message != "" {
		return fmt.Sprintf("goml: %s failed to converge after %d iterations: %s", e.ModelName, e.Iterations, e.Message)
	}
	return fmt.Sprintf("goml: %s failed to converge after %d iterations", e.ModelName, e.Iterations)
}

// ============================================================================
// 共通エラー定義（プロジェクト全体で使用）
// ============================================================================

var (
	// ErrNotFound はリソースが見つからない場合のエラー
	ErrNotFound = errors.New("not found")

	// ErrInvalidArgument は無効な引数が渡された場合のエラー
	ErrInvalidArgument = errors.New("invalid argument")

	// ErrInternal は内部エラーが発生した場合のエラー
	ErrInternal = errors.New("internal error")

	// ErrNotImplemented は未実装の機能を呼び出した場合のエラー
	ErrNotImplemented = errors.New("not implemented")
	
	// 機械学習関連の一般的なエラー定義
	
	// ErrNotFitted はモデルが未学習の場合のエラー
	ErrNotFitted = errors.New("model is not fitted")
	
	// ErrDimensionMismatch は次元が一致しない場合のエラー
	ErrDimensionMismatch = errors.New("dimension mismatch")
	
	// ErrEmptyData は空のデータが渡された場合のエラー
	ErrEmptyData = errors.New("empty data")
	
	// ErrInvalidParameter は無効なパラメータが渡された場合のエラー
	ErrInvalidParameter = errors.New("invalid parameter")
	
	// ErrSingularMatrix は特異行列の場合のエラー
	ErrSingularMatrix = errors.New("singular matrix")
)

// ============================================================================
// エラー生成関数（スタックトレース付き）
// ============================================================================

// NewModelError は新しいModelErrorを作成する（スタックトレース付き）
func NewModelError(op, kind string, err error) error {
	modelErr := &ModelError{
		Op:   op,
		Kind: kind,
		Err:  err,
	}
	
	// cockroachdb/errorsでラップしてスタックトレースを付与
	wrappedErr := errors.WithStack(modelErr)
	
	// デバッグ情報を追加
	if err != nil {
		wrappedErr = errors.WithDetail(wrappedErr, fmt.Sprintf("Operation: %s, Kind: %s, Original error: %v", op, kind, err))
	} else {
		wrappedErr = errors.WithDetail(wrappedErr, fmt.Sprintf("Operation: %s, Kind: %s", op, kind))
	}
	
	return wrappedErr
}

// NewDimensionError は新しいDimensionErrorを作成する（スタックトレース付き）
func NewDimensionError(op string, expected, got []int) error {
	dimErr := &DimensionError{
		Op:       op,
		Expected: expected,
		Got:      got,
	}
	
	// cockroachdb/errorsでラップしてスタックトレースを付与
	wrappedErr := errors.WithStack(dimErr)
	
	// デバッグ情報を追加
	wrappedErr = errors.WithDetail(wrappedErr, fmt.Sprintf("Operation: %s", op))
	wrappedErr = errors.WithDetail(wrappedErr, fmt.Sprintf("Expected dimensions: %v", expected))
	wrappedErr = errors.WithDetail(wrappedErr, fmt.Sprintf("Got dimensions: %v", got))
	wrappedErr = errors.WithHint(wrappedErr, "Check that input dimensions match expected dimensions")
	
	return wrappedErr
}

// NewNotFittedError は新しいNotFittedErrorを作成する（スタックトレース付き）
func NewNotFittedError(modelName string) error {
	notFittedErr := &NotFittedError{
		ModelName: modelName,
	}
	
	// cockroachdb/errorsでラップしてスタックトレースを付与
	wrappedErr := errors.WithStack(notFittedErr)
	
	// デバッグ情報を追加
	wrappedErr = errors.WithDetail(wrappedErr, fmt.Sprintf("Model: %s", modelName))
	wrappedErr = errors.WithHint(wrappedErr, fmt.Sprintf("Call Fit() method on %s before calling Predict()", modelName))
	
	return wrappedErr
}

// NewValueError は新しいValueErrorを作成する（スタックトレース付き）
func NewValueError(op, param string, value interface{}, message string) error {
	valErr := &ValueError{
		Op:      op,
		Param:   param,
		Value:   value,
		Message: message,
	}
	
	// cockroachdb/errorsでラップしてスタックトレースを付与
	wrappedErr := errors.WithStack(valErr)
	
	// デバッグ情報を追加
	wrappedErr = errors.WithDetail(wrappedErr, fmt.Sprintf("Operation: %s", op))
	wrappedErr = errors.WithDetail(wrappedErr, fmt.Sprintf("Parameter: %s", param))
	wrappedErr = errors.WithDetail(wrappedErr, fmt.Sprintf("Value: %v", value))
	if message != "" {
		wrappedErr = errors.WithDetail(wrappedErr, fmt.Sprintf("Message: %s", message))
	}
	
	return wrappedErr
}

// NewConvergenceError は新しいConvergenceErrorを作成する（スタックトレース付き）
func NewConvergenceError(modelName string, iterations int, message string) error {
	convErr := &ConvergenceError{
		ModelName:  modelName,
		Iterations: iterations,
		Message:    message,
	}
	
	// cockroachdb/errorsでラップしてスタックトレースを付与
	wrappedErr := errors.WithStack(convErr)
	
	// デバッグ情報を追加
	wrappedErr = errors.WithDetail(wrappedErr, fmt.Sprintf("Model: %s", modelName))
	wrappedErr = errors.WithDetail(wrappedErr, fmt.Sprintf("Iterations: %d", iterations))
	if message != "" {
		wrappedErr = errors.WithDetail(wrappedErr, fmt.Sprintf("Message: %s", message))
	}
	wrappedErr = errors.WithHint(wrappedErr, "Try increasing max_iterations or adjusting convergence criteria")
	
	return wrappedErr
}