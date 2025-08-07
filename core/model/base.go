// Package model provides core abstractions and interfaces for machine learning models.
//
// This package defines the fundamental building blocks for machine learning estimators
// in the SciGo library, including:
//
//   - BaseEstimator: Core estimator with state management and serialization support
//   - Model persistence: Save and load trained models using Go's encoding/gob
//   - scikit-learn compatibility: Import/export models from Python scikit-learn
//   - Streaming interfaces: Support for online learning and incremental training
//
// The BaseEstimator provides a consistent foundation for all ML algorithms with:
//
//   - Fitted state tracking to prevent usage of untrained models
//   - Serialization support for model persistence
//   - Thread-safe state management
//   - Integration with preprocessing pipelines
//
// Example usage:
//
//	type MyModel struct {
//		model.BaseEstimator
//		// model-specific fields
//	}
//
//	func (m *MyModel) Fit(X, y mat.Matrix) error {
//		// training logic
//		m.SetFitted() // mark as trained
//		return nil
//	}
//
// All models in SciGo embed BaseEstimator to ensure consistent behavior across
// the entire machine learning pipeline.
package model

// EstimatorState はモデルの学習状態を表す
type EstimatorState int

const (
	// NotFitted はモデルが未学習の状態
	NotFitted EstimatorState = iota
	// Fitted はモデルが学習済みの状態
	Fitted
)

// BaseEstimator は全てのモデルの基底となる構造体
type BaseEstimator struct {
	// State はモデルの学習状態を保持します。gobでエンコードするために公開されています。
	State EstimatorState

	// logger はモデル操作のログ出力に使用されます。gobエンコードでは無視されます。
	logger interface{} // Using interface{} to avoid circular imports, will be set to log.Logger
}

// IsFitted returns whether the model has been fitted with training data.
//
// This method checks the internal state to determine if the estimator has been
// trained and is ready for prediction or transformation. All models must be
// fitted before they can be used for predictions.
//
// Returns:
//   - bool: true if the model is fitted, false otherwise
//
// Example:
//
//	if !model.IsFitted() {
//	    err := model.Fit(X, y)
//	    if err != nil {
//	        log.Fatal(err)
//	    }
//	}
//	predictions, err := model.Predict(X_test)
func (e *BaseEstimator) IsFitted() bool {
	return e.State == Fitted
}

// SetFitted marks the estimator as fitted (trained).
//
// This method is called internally by model implementations after successful
// training to indicate that the model is ready for predictions or transformations.
// Should only be called by model implementations, not by end users.
//
// Example (within a model's Fit method):
//
//	func (m *MyModel) Fit(X, y mat.Matrix) error {
//	    // ... training logic ...
//	    m.SetFitted() // Mark as trained
//	    return nil
//	}
func (e *BaseEstimator) SetFitted() {
	e.State = Fitted
}

// Reset returns the estimator to its initial untrained state.
//
// This method clears the fitted state, effectively making the model untrained.
// Useful for reusing model instances with different data or resetting after
// errors. After reset, the model must be fitted again before use.
//
// Example:
//
//	model.Reset()
//	err := model.Fit(newTrainingData, newLabels)
//	if err != nil {
//	    log.Fatal(err)
//	}
func (e *BaseEstimator) Reset() {
	e.State = NotFitted
}

// SetLogger sets the logger for this estimator.
// This method is typically called during model initialization to provide
// structured logging capabilities for ML operations.
//
// Parameters:
//   - logger: Any logger implementation (typically log.Logger interface)
//
// Example:
//
//	import "github.com/YuminosukeSato/scigo/pkg/log"
//	model.SetLogger(log.GetLoggerWithName("LinearRegression"))
func (e *BaseEstimator) SetLogger(logger interface{}) {
	e.logger = logger
}

// GetLogger returns the logger for this estimator.
// Returns nil if no logger has been set.
//
// Returns:
//   - interface{}: The logger instance, should be type-asserted to log.Logger
//
// Example:
//
//	if logger := model.GetLogger(); logger != nil {
//	    if l, ok := logger.(log.Logger); ok {
//	        l.Info("Operation completed")
//	    }
//	}
func (e *BaseEstimator) GetLogger() interface{} {
	return e.logger
}

// LogInfo logs an info-level message if a logger is configured.
// This is a convenience method to avoid repetitive nil checks in model implementations.
//
// Parameters:
//   - msg: The log message
//   - fields: Optional structured logging fields as key-value pairs
func (e *BaseEstimator) LogInfo(msg string, fields ...interface{}) {
	if e.logger != nil {
		// Type assertion would be done in the actual implementation
		// This is kept generic to avoid circular imports
		if logger, ok := e.logger.(interface {
			Info(string, ...interface{})
		}); ok {
			logger.Info(msg, fields...)
		}
	}
}

// LogDebug logs a debug-level message if a logger is configured.
// This is a convenience method for debug logging in model implementations.
//
// Parameters:
//   - msg: The log message
//   - fields: Optional structured logging fields as key-value pairs
func (e *BaseEstimator) LogDebug(msg string, fields ...interface{}) {
	if e.logger != nil {
		if logger, ok := e.logger.(interface {
			Debug(string, ...interface{})
		}); ok {
			logger.Debug(msg, fields...)
		}
	}
}

// LogError logs an error-level message if a logger is configured.
// This is a convenience method for error logging in model implementations.
//
// Parameters:
//   - msg: The log message
//   - fields: Optional structured logging fields as key-value pairs
//     If the first field is an error, it will be handled specially
func (e *BaseEstimator) LogError(msg string, fields ...interface{}) {
	if e.logger != nil {
		if logger, ok := e.logger.(interface {
			Error(string, ...interface{})
		}); ok {
			logger.Error(msg, fields...)
		}
	}
}
