// Package log defines standard attribute keys for machine learning operations.
//
// This file contains predefined attribute keys that provide consistency across
// all logging operations in SciGo. Using these standard keys enables better
// log analysis, monitoring, and debugging of machine learning workflows.
//
// The attributes are organized into categories:
//   - Model and Operation Context
//   - Data Shape and Characteristics
//   - Performance Metrics
//   - Error Context
//
// These keys follow a hierarchical naming convention (e.g., "model.name",
// "data.samples") to enable structured log analysis and filtering.

package log

// Model and Operation Context
// These attributes identify the model type, instance, and operation being performed.
const (
	// ModelNameKey identifies the type of machine learning model.
	// Examples: "LinearRegression", "StandardScaler", "RandomForest"
	ModelNameKey = "model.name"

	// EstimatorIDKey provides a unique identifier for a specific model instance.
	// This is useful for tracking multiple instances of the same model type.
	// Examples: "lr-001", "scaler-abc123", UUID strings
	EstimatorIDKey = "estimator.id"

	// OperationKey specifies the machine learning operation being performed.
	// Standard values: "fit", "predict", "transform", "fit_transform", "score"
	OperationKey = "ml.operation"

	// ComponentKey identifies which component or package is performing the operation.
	// Examples: "linear", "preprocessing", "metrics"
	ComponentKey = "ml.component"

	// PhaseKey indicates the phase of model lifecycle.
	// Examples: "training", "inference", "validation", "preprocessing"
	PhaseKey = "ml.phase"
)

// Data Shape and Characteristics
// These attributes describe the structure and properties of data being processed.
const (
	// SamplesKey indicates the number of samples (rows) in the dataset.
	// This is crucial for understanding the scale of data being processed.
	SamplesKey = "data.samples"

	// FeaturesKey indicates the number of features (columns) in the dataset.
	// Important for dimensionality tracking and debugging shape mismatches.
	FeaturesKey = "data.features"

	// TargetsKey indicates the number of target variables for supervised learning.
	// Usually 1 for single-target problems, >1 for multi-target problems.
	TargetsKey = "data.targets"

	// DataTypeKey specifies the type of data being processed.
	// Examples: "float64", "int32", "categorical", "mixed"
	DataTypeKey = "data.type"

	// DataSizeKey indicates the memory size of the data in bytes.
	// Useful for memory usage monitoring and optimization.
	DataSizeKey = "data.size_bytes"

	// BatchSizeKey indicates the size of processing batches.
	// Relevant for streaming or mini-batch processing scenarios.
	BatchSizeKey = "data.batch_size"
)

// Performance Metrics
// These attributes capture timing, accuracy, and resource usage information.
const (
	// DurationMsKey records the execution time of an operation in milliseconds.
	// This is essential for performance monitoring and optimization.
	DurationMsKey = "perf.duration_ms"

	// DurationSecondsKey records the execution time in seconds for longer operations.
	// Useful for training operations that take minutes or hours.
	DurationSecondsKey = "perf.duration_seconds"

	// MemoryUsageKey records memory usage in bytes during the operation.
	// Important for memory optimization and resource planning.
	MemoryUsageKey = "perf.memory_bytes"

	// AccuracyKey records model accuracy for evaluation operations.
	// Range typically [0.0, 1.0] for classification accuracy.
	AccuracyKey = "metrics.accuracy"

	// LossKey records loss value during training or evaluation.
	// Lower values typically indicate better model performance.
	LossKey = "metrics.loss"

	// R2ScoreKey records R² coefficient of determination for regression.
	// Range typically [-∞, 1.0], with 1.0 being perfect prediction.
	R2ScoreKey = "metrics.r2_score"

	// IterationKey records the current iteration number during iterative processes.
	// Useful for tracking convergence in iterative algorithms.
	IterationKey = "training.iteration"

	// EpochKey records the current epoch number during training.
	// Standard in neural networks and iterative learning algorithms.
	EpochKey = "training.epoch"
)

// Prediction and Output Context
// These attributes describe prediction operations and their results.
const (
	// PredsKey indicates the number of predictions made.
	// Useful for throughput monitoring and batch size optimization.
	PredsKey = "preds.count"

	// PredsBatchKey indicates the batch number for prediction operations.
	// Relevant for streaming or large-scale batch prediction scenarios.
	PredsBatchKey = "preds.batch"

	// ConfidenceKey records prediction confidence or probability.
	// Range typically [0.0, 1.0] for classification confidence.
	ConfidenceKey = "preds.confidence"

	// ThresholdKey records decision thresholds used for classification.
	// Important for understanding model decision boundaries.
	ThresholdKey = "preds.threshold"
)

// Error and Warning Context
// These attributes provide additional context for error and warning messages.
const (
	// ErrorCodeKey provides a structured error code for programmatic handling.
	// Examples: "DIMENSION_MISMATCH", "NOT_FITTED", "CONVERGENCE_FAILURE"
	ErrorCodeKey = "error.code"

	// ErrorTypeKey categorizes the type of error encountered.
	// Examples: "ValidationError", "ConvergenceError", "DataError"
	ErrorTypeKey = "error.type"

	// StacktraceKey contains stack trace information for debugging.
	// Automatically populated by the error logging functions.
	StacktraceKey = "error.stacktrace"

	// SuggestionKey provides helpful suggestions for resolving issues.
	// Examples: "Check input data shape", "Increase max_iterations"
	SuggestionKey = "error.suggestion"
)

// Hyperparameters and Configuration
// These attributes capture model configuration and hyperparameters.
const (
	// HyperParamsKey contains model hyperparameters as a structured object.
	// Useful for tracking model configuration and reproducibility.
	HyperParamsKey = "model.hyperparams"

	// LearningRateKey records the learning rate for gradient-based algorithms.
	// Critical hyperparameter for training stability and convergence.
	LearningRateKey = "hyperparams.learning_rate"

	// RegularizationKey records regularization strength (L1, L2, etc.).
	// Important for understanding model complexity and overfitting prevention.
	RegularizationKey = "hyperparams.regularization"

	// RandomSeedKey records the random seed for reproducibility.
	// Essential for debugging and ensuring reproducible results.
	RandomSeedKey = "config.random_seed"

	// ConfigVersionKey tracks configuration or model version.
	// Useful for A/B testing and model versioning.
	ConfigVersionKey = "config.version"
)

// Infrastructure and Environment
// These attributes describe the execution environment and resource usage.
const (
	// HostnameKey identifies the machine or container running the operation.
	// Useful for distributed systems and debugging deployment issues.
	HostnameKey = "infra.hostname"

	// ProcessIDKey records the process ID for the operation.
	// Helpful for debugging and resource tracking.
	ProcessIDKey = "infra.pid"

	// ThreadIDKey records the thread or goroutine ID.
	// Useful for concurrent processing debugging.
	ThreadIDKey = "infra.thread_id"

	// GPUIDKey identifies which GPU device is being used (if applicable).
	// Important for GPU resource management and debugging.
	GPUIDKey = "infra.gpu_id"

	// WorkerIDKey identifies worker processes in distributed systems.
	// Relevant for parameter servers and distributed training.
	WorkerIDKey = "infra.worker_id"
)

// Standard attribute value constants for common operations.
// Using these constants ensures consistency across the codebase.
const (
	// Standard ML operations
	OperationFit          = "fit"
	OperationPredict      = "predict"
	OperationTransform    = "transform"
	OperationFitTransform = "fit_transform"
	OperationScore        = "score"
	OperationPartialFit   = "partial_fit"

	// Standard ML phases
	PhaseTraining      = "training"
	PhaseValidation    = "validation"
	PhaseTesting       = "testing"
	PhaseInference     = "inference"
	PhasePreprocessing = "preprocessing"

	// Standard error codes
	ErrorNotFitted         = "NOT_FITTED"
	ErrorDimensionMismatch = "DIMENSION_MISMATCH"
	ErrorEmptyData         = "EMPTY_DATA"
	ErrorInvalidInput      = "INVALID_INPUT"
	ErrorConvergence       = "CONVERGENCE_FAILURE"
	ErrorSingularMatrix    = "SINGULAR_MATRIX"
)
