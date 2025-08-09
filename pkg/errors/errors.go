// Package errors provides error handling and warning systems for the entire project.
// Inspired by scikit-learn's warning and exception system, it provides structured error information.
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
//	Global Warning Handling
//
// ===========================================================================
var (
	warningMutex   sync.Mutex
	warningHandler = func(w error) {
		// Default handler logs to standard error output
		log.Printf("GoML-Warning: %v\n", w)
	}
	// zerolog logger (lazy initialization to avoid circular import)
	zerologWarnFunc func(warning error)
)

// SetWarningHandler sets the warning handler for the entire GoML library.
// This allows you to control how custom warnings like ConvergenceWarning are handled.
//
// Example:
//
//	errors.SetWarningHandler(func(w error) {
//	    // Ignore warnings
//	})
func SetWarningHandler(handler func(w error)) {
	warningMutex.Lock()
	defer warningMutex.Unlock()
	warningHandler = handler
}

// SetZerologWarnFunc sets the zerolog warning function (to avoid circular import).
func SetZerologWarnFunc(warnFunc func(warning error)) {
	warningMutex.Lock()
	defer warningMutex.Unlock()
	zerologWarnFunc = warnFunc
}

// Warn raises a warning.
// If zerolog is available, it outputs as structured log, otherwise uses traditional handler.
func Warn(w error) {
	warningMutex.Lock()
	defer warningMutex.Unlock()

	// Use zerolog if configured
	if zerologWarnFunc != nil {
		zerologWarnFunc(w)
		return
	}

	// Fallback: traditional handler
	if warningHandler != nil {
		warningHandler(w)
	}
}

// ===========================================================================
//
//	scikit-learn Compatible Warning Types
//
// ===========================================================================

// ConvergenceWarning is a warning raised when optimization algorithms fail to converge.
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

// MarshalZerologObject adds structured warning information to zerolog events.
func (w *ConvergenceWarning) MarshalZerologObject(e *zerolog.Event) {
	e.Str("algorithm", w.Algorithm).
		Int("iterations", w.Iterations).
		Str("message", w.Message).
		Str("type", "ConvergenceWarning")
}

// NewConvergenceWarning creates a new ConvergenceWarning.
func NewConvergenceWarning(algorithm string, iterations int, message string) *ConvergenceWarning {
	return &ConvergenceWarning{Algorithm: algorithm, Iterations: iterations, Message: message}
}

// DataConversionWarning is a warning raised when data types are implicitly converted.
type DataConversionWarning struct {
	FromType string
	ToType   string
	Reason   string
}

func (w *DataConversionWarning) Error() string {
	return fmt.Sprintf("data converted from %s to %s. Reason: %s", w.FromType, w.ToType, w.Reason)
}

// MarshalZerologObject adds structured warning information to zerolog events.
func (w *DataConversionWarning) MarshalZerologObject(e *zerolog.Event) {
	e.Str("from_type", w.FromType).
		Str("to_type", w.ToType).
		Str("reason", w.Reason).
		Str("type", "DataConversionWarning")
}

// NewDataConversionWarning creates a new DataConversionWarning.
func NewDataConversionWarning(from, to, reason string) *DataConversionWarning {
	return &DataConversionWarning{FromType: from, ToType: to, Reason: reason}
}

// UndefinedMetricWarning is a warning raised when metrics cannot be calculated.
// For example, when calculating precision with no positive class predictions.
type UndefinedMetricWarning struct {
	Metric    string
	Condition string
	Result    float64 // Value returned under this condition
}

func (w *UndefinedMetricWarning) Error() string {
	return fmt.Sprintf("'%s' is ill-defined and being set to %f due to %s.", w.Metric, w.Result, w.Condition)
}

// NewUndefinedMetricWarning creates a new UndefinedMetricWarning.
func NewUndefinedMetricWarning(metric, condition string, result float64) *UndefinedMetricWarning {
	return &UndefinedMetricWarning{Metric: metric, Condition: condition, Result: result}
}

// ===========================================================================
//
//	Structured Error Types
//
// ===========================================================================

// NotFittedError is an error raised when calling `Predict` or `Transform` on an unfitted model.
type NotFittedError struct {
	ModelName string
	Method    string
}

func (e *NotFittedError) Error() string {
	return fmt.Sprintf("goml: %s: this model is not fitted yet. Call Fit() before using %s()", e.ModelName, e.Method)
}

// MarshalZerologObject adds structured error information to zerolog events.
func (e *NotFittedError) MarshalZerologObject(event *zerolog.Event) {
	event.Str("model_name", e.ModelName).
		Str("method", e.Method).
		Str("type", "NotFittedError")
}

// NewNotFittedError creates a new NotFittedError with stack trace.
func NewNotFittedError(modelName, method string) error {
	err := &NotFittedError{ModelName: modelName, Method: method}
	return errors.WithStack(err)
}

// DimensionError is an error raised when input data dimensions don't match expected values.
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

// MarshalZerologObject adds structured error information to zerolog events.
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

// NewDimensionError creates a new DimensionError with stack trace.
func NewDimensionError(op string, expected, got, axis int) error {
	err := &DimensionError{Op: op, Expected: expected, Got: got, Axis: axis}
	return errors.WithStack(err)
}

// ValidationError is an error raised when input parameter validation fails.
// It indicates more specific validation logic failures than `ValueError`.
type ValidationError struct {
	ParamName string
	Reason    string
	Value     interface{}
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("goml: validation failed for parameter '%s': %s (got: %v)", e.ParamName, e.Reason, e.Value)
}

// MarshalZerologObject adds structured error information to zerolog events.
func (e *ValidationError) MarshalZerologObject(event *zerolog.Event) {
	event.Str("param_name", e.ParamName).
		Str("reason", e.Reason).
		Interface("value", e.Value).
		Str("type", "ValidationError")
}

// NewValidationError creates a new ValidationError with stack trace.
func NewValidationError(param, reason string, value interface{}) error {
	err := &ValidationError{ParamName: param, Reason: reason, Value: value}
	return errors.WithStack(err)
}

// ValueError is an error raised when argument values are inappropriate or invalid.
// For example, passing negative numbers to a `log` function.
type ValueError struct {
	Op      string
	Message string
}

func (e *ValueError) Error() string {
	return fmt.Sprintf("goml: %s: %s", e.Op, e.Message)
}

// NewValueError creates a new ValueError with stack trace.
func NewValueError(op, message string) error {
	err := &ValueError{Op: op, Message: message}
	return errors.WithStack(err)
}

// ModelError is a general error related to machine learning models.
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

// NewModelError creates a new ModelError with stack trace.
func NewModelError(op, kind string, err error) error {
	modelErr := &ModelError{Op: op, Kind: kind, Err: err}
	return errors.WithStack(modelErr)
}

// ===========================================================================
//
//	cockroachdb/errors Wrapper Functions
//
// ===========================================================================

// Is reports whether err matches target.
func Is(err, target error) bool {
	return errors.Is(err, target)
}

// As finds the first error in err's chain that matches target.
func As(err error, target interface{}) bool {
	return errors.As(err, target)
}

// Wrap wraps an existing error with a message.
func Wrap(err error, message string) error {
	return errors.Wrap(err, message)
}

// Wrapf wraps an existing error with a formatted message.
func Wrapf(err error, format string, args ...interface{}) error {
	return errors.Wrapf(err, format, args...)
}

// New creates a new error.
func New(message string) error {
	return errors.New(message)
}

// Newf creates a new formatted error.
func Newf(format string, args ...interface{}) error {
	return errors.Newf(format, args...)
}

// WithStack annotates err with a stack trace.
func WithStack(err error) error {
	return errors.WithStack(err)
}

// ===========================================================================
//
//	Online Learning Specific Error Types
//
// ===========================================================================

// NumericalInstabilityError is an error raised when numerical computation becomes unstable.
// It detects NaN, Inf, overflow, underflow, etc.
type NumericalInstabilityError struct {
	Operation string                 // Operation where error occurred (e.g., "gradient_update", "loss_calculation")
	Values    []float64              // Problematic values
	Context   map[string]interface{} // Additional context for debugging
	Iteration int                    // Iteration number where error occurred
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

// NewNumericalInstabilityError creates a new NumericalInstabilityError.
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
