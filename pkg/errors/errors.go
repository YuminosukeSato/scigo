// Package errors はプロジェクト全体のエラーハンドリングと警告システムを提供します。
// scikit-learnの警告・例外システムにインスパイアされており、構造化されたエラー情報を提供します。
package errors

import (
	"fmt"
	"log"
	"sync"

	"github.com/cockroachdb/errors"
)

// ===========================================================================
//
//	グローバル警告ハンドリング
//
// ===========================================================================
var (
	warningMutex   sync.Mutex
	warningHandler = func(w error) {
		// デフォルトのハンドラは標準エラー出力にログを出す
		log.Printf("GoML-Warning: %v\n", w)
	}
)

// SetWarningHandler はGoMLライブラリ全体の警告ハンドラを設定します。
// これにより、ConvergenceWarningなどのカスタム警告の処理方法を制御できます。
//
// 例:
//
//	errors.SetWarningHandler(func(w error) {
//	    // 警告を無視する
//	})
func SetWarningHandler(handler func(w error)) {
	warningMutex.Lock()
	defer warningMutex.Unlock()
	warningHandler = handler
}

// Warn は警告を発生させます。
// 設定された警告ハンドラを呼び出します。
func Warn(w error) {
	warningMutex.Lock()
	defer warningMutex.Unlock()
	if warningHandler != nil {
		warningHandler(w)
	}
}

// ===========================================================================
//
//	scikit-learn互換の警告型
//
// ===========================================================================

// ConvergenceWarning は最適化アルゴリズムが収束しなかった場合に発生する警告です。
type ConvergenceWarning struct {
	Algorithm  string
	Iterations int
	Message    string
}

func (w *ConvergenceWarning) Error() string {
	if w.Message != "" {
		return fmt.Sprintf("%s failed to converge after %d iterations: %s", w.Algorithm, w.Iterations, w.Message)
	}
	return fmt.Sprintf("%s failed to converge after %d iterations. Consider increasing max_iter or adjusting parameters.", w.Algorithm, w.Iterations)
}

// NewConvergenceWarning は新しいConvergenceWarningを作成します。
func NewConvergenceWarning(algorithm string, iterations int, message string) *ConvergenceWarning {
	return &ConvergenceWarning{Algorithm: algorithm, Iterations: iterations, Message: message}
}

// DataConversionWarning はデータの型が暗黙的に変換された場合に発生する警告です。
type DataConversionWarning struct {
	FromType string
	ToType   string
	Reason   string
}

func (w *DataConversionWarning) Error() string {
	return fmt.Sprintf("data converted from %s to %s. Reason: %s", w.FromType, w.ToType, w.Reason)
}

// NewDataConversionWarning は新しいDataConversionWarningを作成します。
func NewDataConversionWarning(from, to, reason string) *DataConversionWarning {
	return &DataConversionWarning{FromType: from, ToType: to, Reason: reason}
}

// UndefinedMetricWarning は評価指標が計算できない場合に発生する警告です。
// 例えば、適合率(precision)を計算する際に、陽性クラスの予測が一つもなかった場合など。
type UndefinedMetricWarning struct {
	Metric     string
	Condition  string
	Result     float64 // この条件で返される値
}

func (w *UndefinedMetricWarning) Error() string {
	return fmt.Sprintf("'%s' is ill-defined and being set to %f due to %s.", w.Metric, w.Result, w.Condition)
}

// NewUndefinedMetricWarning は新しいUndefinedMetricWarningを作成します。
func NewUndefinedMetricWarning(metric, condition string, result float64) *UndefinedMetricWarning {
	return &UndefinedMetricWarning{Metric: metric, Condition: condition, Result: result}
}

// ===========================================================================
//
//	構造化されたエラー型
//
// ===========================================================================

// NotFittedError はモデルが未学習の状態で `Predict` や `Transform` を呼び出した場合のエラーです。
type NotFittedError struct {
	ModelName string
	Method    string
}

func (e *NotFittedError) Error() string {
	return fmt.Sprintf("goml: %s: this model is not fitted yet. Call Fit() before using %s()", e.ModelName, e.Method)
}

// NewNotFittedError は新しいNotFittedErrorを作成し、スタックトレースを付与します。
func NewNotFittedError(modelName, method string) error {
	err := &NotFittedError{ModelName: modelName, Method: method}
	return errors.WithStack(err)
}

// DimensionError は入力データの次元が期待値と異なる場合のエラーです。
type DimensionError struct {
	Op       string
	Expected int
	Got      int
	Axis     int // 0 for rows, 1 for columns/features
}

func (e *DimensionError) Error() string {
	axisName := "features"
	if e.Axis == 0 {
		axisName = "rows"
	}
	return fmt.Sprintf("goml: %s: dimension mismatch on axis %d (%s). Expected %d, got %d", e.Op, e.Axis, axisName, e.Expected, e.Got)
}

// NewDimensionError は新しいDimensionErrorを作成し、スタックトレースを付与します。
func NewDimensionError(op string, expected, got, axis int) error {
	err := &DimensionError{Op: op, Expected: expected, Got: got, Axis: axis}
	return errors.WithStack(err)
}

// ValidationError は入力パラメータの検証に失敗した場合のエラーです。
// `ValueError`よりも具体的なバリデーションロジックの失敗を示します。
type ValidationError struct {
	ParamName string
	Reason    string
	Value     interface{}
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("goml: validation failed for parameter '%s': %s (got: %v)", e.ParamName, e.Reason, e.Value)
}

// NewValidationError は新しいValidationErrorを作成し、スタックトレースを付与します。
func NewValidationError(param, reason string, value interface{}) error {
	err := &ValidationError{ParamName: param, Reason: reason, Value: value}
	return errors.WithStack(err)
}

// ValueError は引数の値が不適切または不正な場合に発生するエラーです。
// 例えば、`log`関数に負の数を渡した場合など。
type ValueError struct {
	Op      string
	Message string
}

func (e *ValueError) Error() string {
	return fmt.Sprintf("goml: %s: %s", e.Op, e.Message)
}

// NewValueError は新しいValueErrorを作成し、スタックトレースを付与します。
func NewValueError(op, message string) error {
	err := &ValueError{Op: op, Message: message}
	return errors.WithStack(err)
}

// ModelError は機械学習モデルに関する一般的なエラーです。
type ModelError struct {
	Op   string
	Kind string
	Err  error
}

func (e *ModelError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("goml: %s: %s: %v", e.Op, e.Kind, e.Err)
	}
	return fmt.Sprintf("goml: %s: %s", e.Op, e.Kind)
}

func (e *ModelError) Unwrap() error {
	return e.Err
}

// NewModelError は新しいModelErrorを作成し、スタックトレースを付与します。
func NewModelError(op, kind string, err error) error {
	modelErr := &ModelError{Op: op, Kind: kind, Err: err}
	return errors.WithStack(modelErr)
}

// ===========================================================================
//
//	cockroachdb/errors ラッパー関数
//
// ===========================================================================

// Is はエラーが特定のターゲットエラーかどうかを判定します。
func Is(err, target error) bool {
	return errors.Is(err, target)
}

// As はエラーが特定の型にキャスト可能かどうかを判定します。
func As(err error, target interface{}) bool {
	return errors.As(err, target)
}

// Wrap は既存のエラーをメッセージ付きでラップします。
func Wrap(err error, message string) error {
	return errors.Wrap(err, message)
}

// Wrapf は既存のエラーをフォーマット文字列でラップします。
func Wrapf(err error, format string, args ...interface{}) error {
	return errors.Wrapf(err, format, args...)
}

// New は新しいエラーを作成します。
func New(message string) error {
	return errors.New(message)
}

// Newf は新しいフォーマット済みエラーを作成します。
func Newf(format string, args ...interface{}) error {
	return errors.Newf(format, args...)
}

// WithStack はエラーにスタックトレースを付与します。
func WithStack(err error) error {
	return errors.WithStack(err)
}

// ===========================================================================
//
//	共通エラー変数
//
// ===========================================================================

var (
	// ErrNotImplemented は機能が未実装の場合のエラーです。
	ErrNotImplemented = New("not implemented")
	
	// ErrEmptyData は空のデータが渡された場合のエラーです。
	ErrEmptyData = New("empty data")
	
	// ErrSingularMatrix は特異行列の場合のエラーです。
	ErrSingularMatrix = New("singular matrix")
)
