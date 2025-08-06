// Package log provides a structured logging interface for SciGo machine learning operations.
//
// This package defines a minimal, slog-compatible logging interface that allows for
// flexible implementation switching while providing ML-specific structured logging
// capabilities. The interface is designed to integrate seamlessly with Go's standard
// log/slog package and popular logging libraries like zerolog, logrus, and zap.
//
// Key features:
//   - slog-compatible interface for future-proofing
//   - ML-specific structured attributes (operation types, data shapes, metrics)
//   - Context-aware logging with field chaining
//   - Performance-optimized with lazy evaluation support
//   - Test-friendly with configurable output destinations
//
// Example usage:
//   logger := log.GetLogger().With(
//       log.ModelNameKey, "LinearRegression",
//       log.EstimatorIDKey, "lr-001",
//   )
//   logger.Info("Training started",
//       log.OperationKey, "fit",
//       log.SamplesKey, 1000,
//       log.FeaturesKey, 5,
//   )

package log

import (
	"context"
)

// Logger defines a structured logging interface compatible with Go's log/slog.
//
// This interface provides the core logging methods with structured field support,
// allowing for rich contextual information to be included with log messages.
// It's designed to be implementation-agnostic, enabling easy switching between
// different logging backends while maintaining a consistent API.
//
// The interface supports method chaining through the With method, allowing
// for creation of contextual loggers with pre-populated fields.
type Logger interface {
	// Debug logs a debug-level message with optional structured fields.
	// Debug logs are typically used for detailed diagnostic information
	// and are usually disabled in production environments.
	//
	// Parameters:
	//   - msg: The primary log message
	//   - fields: Optional key-value pairs for structured logging
	//
	// Example:
	//   logger.Debug("Processing data batch",
	//       "batch_id", 42,
	//       "size", 100,
	//   )
	Debug(msg string, fields ...any)

	// Info logs an info-level message with optional structured fields.
	// Info logs are used for general operational information about
	// the application's execution flow.
	//
	// Parameters:
	//   - msg: The primary log message
	//   - fields: Optional key-value pairs for structured logging
	//
	// Example:
	//   logger.Info("Model training completed",
	//       log.DurationMsKey, 5432,
	//       log.AccuracyKey, 0.95,
	//   )
	Info(msg string, fields ...any)

	// Warn logs a warning-level message with optional structured fields.
	// Warning logs indicate potentially problematic situations that
	// don't prevent the application from continuing.
	//
	// Parameters:
	//   - msg: The primary log message
	//   - fields: Optional key-value pairs for structured logging
	//
	// Example:
	//   logger.Warn("Model performance below threshold",
	//       log.AccuracyKey, 0.65,
	//       "threshold", 0.8,
	//   )
	Warn(msg string, fields ...any)

	// Error logs an error-level message with optional structured fields.
	// Error logs indicate error conditions that should be investigated.
	// If an error value is provided as the first field, stack trace
	// information may be automatically included.
	//
	// Parameters:
	//   - msg: The primary log message
	//   - fields: Optional key-value pairs for structured logging
	//             If the first field is an error, it will be handled specially
	//
	// Example:
	//   logger.Error("Model training failed",
	//       err,
	//       log.OperationKey, "fit",
	//       log.SamplesKey, 1000,
	//   )
	Error(msg string, fields ...any)

	// With returns a new Logger with the given fields pre-populated.
	// This method enables creation of contextual loggers that automatically
	// include common fields in all subsequent log messages.
	//
	// Parameters:
	//   - fields: Key-value pairs to include in all future log messages
	//
	// Returns:
	//   - Logger: A new logger instance with the specified fields
	//
	// Example:
	//   contextLogger := logger.With(
	//       log.ModelNameKey, "RandomForest",
	//       log.EstimatorIDKey, "rf-123",
	//   )
	//   contextLogger.Info("Starting training")  // Automatically includes model info
	With(fields ...any) Logger

	// Enabled reports whether the logger emits log records at the given level.
	// This method can be used to avoid expensive operations when constructing
	// log messages that won't be emitted.
	//
	// Parameters:
	//   - ctx: Context for the logging operation
	//   - level: The log level to check
	//
	// Returns:
	//   - bool: true if the logger would emit a record at the given level
	//
	// Example:
	//   if logger.Enabled(ctx, LevelDebug) {
	//       expensiveData := calculateExpensiveMetrics()
	//       logger.Debug("Detailed metrics", "metrics", expensiveData)
	//   }
	Enabled(ctx context.Context, level Level) bool
}

// Level represents a logging level, compatible with slog.Level.
// This type allows for level-based filtering of log messages.
type Level int

// Standard logging levels, values are compatible with slog.Level.
const (
	LevelDebug Level = -4  // Detailed diagnostic information
	LevelInfo  Level = 0   // General operational information  
	LevelWarn  Level = 4   // Warning conditions
	LevelError Level = 8   // Error conditions
)

// String returns the string representation of the log level.
func (l Level) String() string {
	switch l {
	case LevelDebug:
		return "DEBUG"
	case LevelInfo:
		return "INFO"
	case LevelWarn:
		return "WARN"
	case LevelError:
		return "ERROR"
	default:
		return "UNKNOWN"
	}
}

// LoggerProvider defines an interface for creating and configuring loggers.
// This interface allows for dependency injection and testing with different
// logger implementations.
type LoggerProvider interface {
	// GetLogger returns the default logger instance.
	GetLogger() Logger

	// GetLoggerWithName returns a logger with a specific name/component identifier.
	GetLoggerWithName(name string) Logger

	// SetLevel sets the minimum log level for all loggers created by this provider.
	SetLevel(level Level)
}