// Package errors はプロジェクト全体のエラーハンドリングと警告システムを提供します。
// scikit-learnの警告・例外システムにインスパイアされており、構造化されたエラー情報を提供します。
package errors

import (
	"fmt"
	"log"
	"sync"

	"github.com/cockroachdb/errors"
	"github.com/rs/zerolog"
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
	// zerologロガー（循環importを避けるため遅延初期化）
	zerologWarnFunc func(warning error)
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

// SetZerologWarnFunc はzerolog警告関数を設定します（循環importを避けるため）。
func SetZerologWarnFunc(warnFunc func(warning error)) {
	warningMutex.Lock()
	defer warningMutex.Unlock()
	zerologWarnFunc = warnFunc
}

// Warn は警告を発生させます。
// zerologが利用可能な場合は構造化ログとして出力し、そうでなければ従来のハンドラを使用します。
func Warn(w error) {
	warningMutex.Lock()
	defer warningMutex.Unlock()

	// zerologが設定されている場合は優先的に使用
	if zerologWarnFunc != nil {
		zerologWarnFunc(w)
		return
	}

	// フォールバック: 従来のハンドラ
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

// MarshalZerologObject はzerologのイベントに構造化された警告情報を追加します。
func (w *ConvergenceWarning) MarshalZerologObject(e *zerolog.Event) {
	e.Str("algorithm", w.Algorithm).
		Int("iterations", w.Iterations).
		Str("message", w.Message).
		Str("type", "ConvergenceWarning")
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

// MarshalZerologObject はzerologのイベントに構造化された警告情報を追加します。
func (w *DataConversionWarning) MarshalZerologObject(e *zerolog.Event) {
	e.Str("from_type", w.FromType).
		Str("to_type", w.ToType).
		Str("reason", w.Reason).
		Str("type", "DataConversionWarning")
}

// NewDataConversionWarning は新しいDataConversionWarningを作成します。
func NewDataConversionWarning(from, to, reason string) *DataConversionWarning {
	return &DataConversionWarning{FromType: from, ToType: to, Reason: reason}
}

// UndefinedMetricWarning は評価指標が計算できない場合に発生する警告です。
// 例えば、適合率(precision)を計算する際に、陽性クラスの予測が一つもなかった場合など。
type UndefinedMetricWarning struct {
	Metric    string
	Condition string
	Result    float64 // この条件で返される値
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

// MarshalZerologObject はzerologのイベントに構造化されたエラー情報を追加します。
func (e *NotFittedError) MarshalZerologObject(event *zerolog.Event) {
	event.Str("model_name", e.ModelName).
		Str("method", e.Method).
		Str("type", "NotFittedError")
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

// MarshalZerologObject はzerologのイベントに構造化されたエラー情報を追加します。
func (e *DimensionError) MarshalZerologObject(event *zerolog.Event) {
	axisName := "features"
	if e.Axis == 0 {
		axisName = "rows"
	}
	event.Str("operation", e.Op).
		Int("expected", e.Expected).
		Int("got", e.Got).
		Int("axis", e.Axis).
		Str("axis_name", axisName).
		Str("type", "DimensionError")
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

// MarshalZerologObject はzerologのイベントに構造化されたエラー情報を追加します。
func (e *ValidationError) MarshalZerologObject(event *zerolog.Event) {
	event.Str("param_name", e.ParamName).
		Str("reason", e.Reason).
		Interface("value", e.Value).
		Str("type", "ValidationError")
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
//	オンライン学習特有のエラー型
//
// ===========================================================================

// NumericalInstabilityError は数値計算が不安定になった場合のエラーです。
// NaN、Inf、オーバーフロー、アンダーフローなどを検出します。
type NumericalInstabilityError struct {
	Operation string                 // 発生した操作（例: "gradient_update", "loss_calculation"）
	Values    []float64              // 問題のある値
	Context   map[string]interface{} // デバッグ用の追加コンテキスト情報
	Iteration int                    // 発生したイテレーション番号
}

func (e *NumericalInstabilityError) Error() string {
	valStr := ""
	for i, v := range e.Values {
		if i > 0 {
			valStr += ", "
		}
		if i >= 5 {
			valStr += "..."
			break
		}
		valStr += fmt.Sprintf("%.6g", v)
	}
	return fmt.Sprintf("goml: numerical instability detected in %s at iteration %d. Values: [%s]",
		e.Operation, e.Iteration, valStr)
}

// NewNumericalInstabilityError は新しいNumericalInstabilityErrorを作成します。
func NewNumericalInstabilityError(operation string, values []float64, iteration int) error {
	err := &NumericalInstabilityError{
		Operation: operation,
		Values:    values,
		Iteration: iteration,
		Context:   make(map[string]interface{}),
	}
	return errors.WithStack(err)
}

// InputShapeError は入力データの形状が期待と異なる場合のエラーです。
// DimensionErrorより詳細で、訓練時と推論時の不整合を検出します。
type InputShapeError struct {
	Phase    string // "training", "prediction", "transform"
	Expected []int  // 期待される形状
	Got      []int  // 実際の形状
	Feature  string // 問題のある特徴量名（オプション）
}

func (e *InputShapeError) Error() string {
	expectedStr := fmt.Sprintf("%v", e.Expected)
	gotStr := fmt.Sprintf("%v", e.Got)
	if e.Feature != "" {
		return fmt.Sprintf("goml: input shape mismatch in %s phase for feature '%s'. Expected shape %s, got %s",
			e.Phase, e.Feature, expectedStr, gotStr)
	}
	return fmt.Sprintf("goml: input shape mismatch in %s phase. Expected shape %s, got %s",
		e.Phase, expectedStr, gotStr)
}

// NewInputShapeError は新しいInputShapeErrorを作成します。
func NewInputShapeError(phase string, expected, got []int) error {
	err := &InputShapeError{
		Phase:    phase,
		Expected: expected,
		Got:      got,
	}
	return errors.WithStack(err)
}

// ModelDriftWarning はモデルドリフトが検出された場合の警告です。
type ModelDriftWarning struct {
	DriftScore float64 // ドリフトスコア（検出器により異なる）
	Threshold  float64 // 閾値
	Detector   string  // 使用したドリフト検出器（例: "DDM", "ADWIN"）
	Action     string  // 推奨アクション（"reset", "alert", "retrain"）
	Timestamp  int64   // ドリフト検出時のタイムスタンプ（Unix時間）
}

func (w *ModelDriftWarning) Error() string {
	return fmt.Sprintf("Model drift detected by %s: score=%.4f (threshold=%.4f). Recommended action: %s",
		w.Detector, w.DriftScore, w.Threshold, w.Action)
}

// NewModelDriftWarning は新しいModelDriftWarningを作成します。
func NewModelDriftWarning(detector string, score, threshold float64, action string) *ModelDriftWarning {
	return &ModelDriftWarning{
		Detector:   detector,
		DriftScore: score,
		Threshold:  threshold,
		Action:     action,
		Timestamp:  0, // Will be set when detected
	}
}

// CatastrophicForgettingWarning は破滅的忘却が発生した可能性がある場合の警告です。
type CatastrophicForgettingWarning struct {
	OldPerformance float64 // 以前のパフォーマンス
	NewPerformance float64 // 現在のパフォーマンス
	DropRate       float64 // 性能低下率
	Metric         string  // 使用したメトリクス（例: "accuracy", "f1_score"）
}

func (w *CatastrophicForgettingWarning) Error() string {
	return fmt.Sprintf("Possible catastrophic forgetting detected: %s dropped from %.4f to %.4f (%.2f%% decrease)",
		w.Metric, w.OldPerformance, w.NewPerformance, w.DropRate*100)
}

// NewCatastrophicForgettingWarning は新しいCatastrophicForgettingWarningを作成します。
func NewCatastrophicForgettingWarning(metric string, oldPerf, newPerf float64) *CatastrophicForgettingWarning {
	dropRate := (oldPerf - newPerf) / oldPerf
	return &CatastrophicForgettingWarning{
		Metric:         metric,
		OldPerformance: oldPerf,
		NewPerformance: newPerf,
		DropRate:       dropRate,
	}
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
