package errors_test

import (
	"errors"
	"fmt"

	scigoErrors "github.com/YuminosukeSato/scigo/pkg/errors"
)

// Example demonstrates Go 1.13+ error wrapping
func Example() {
	// Create a base error
	baseErr := fmt.Errorf("invalid input value")

	// Wrap the error with context using Go 1.13+ syntax
	wrappedErr := fmt.Errorf("model validation failed: %w", baseErr)

	// Further wrap with operation context
	opErr := fmt.Errorf("LinearRegression.Fit: %w", wrappedErr)

	// Use errors.Is to check for specific error types
	if errors.Is(opErr, baseErr) {
		fmt.Println("Found base error in chain")
	}

	// Unwrap errors to get the underlying cause
	unwrapped := errors.Unwrap(opErr)
	fmt.Printf("Unwrapped: %v\n", unwrapped)

	// Output: Found base error in chain
	// Unwrapped: model validation failed: invalid input value
}

// Example_customErrorTypes demonstrates custom error type handling
func Example_customErrorTypes() {
	// Create a custom error using our error constructors
	dimErr := scigoErrors.NewDimensionError("Transform", 5, 3, 1)

	// Wrap it with additional context
	wrappedErr := fmt.Errorf("preprocessing failed: %w", dimErr)

	// Check if error is of specific type using errors.As
	var dimensionErr *scigoErrors.DimensionError
	if errors.As(wrappedErr, &dimensionErr) {
		fmt.Printf("Dimension error: expected %d, got %d\n",
			dimensionErr.Expected, dimensionErr.Got)
	}

	// Output: Dimension error: expected 5, got 3
}

// Example_errorComparison demonstrates error comparison patterns
func Example_errorComparison() {
	// Create different types of errors
	notFittedErr := scigoErrors.NewNotFittedError("LinearRegression", "Predict")
	valueErr := scigoErrors.NewValueError("StandardScaler", "negative values not supported")

	// Create a sentinel error for comparison
	var customErr = errors.New("custom processing error")
	wrappedCustom := fmt.Errorf("operation failed: %w", customErr)

	// Use errors.Is for sentinel error checking
	if errors.Is(wrappedCustom, customErr) {
		fmt.Println("Custom error detected")
	}

	// Use errors.As for type assertions
	var notFitted *scigoErrors.NotFittedError
	if errors.As(notFittedErr, &notFitted) {
		fmt.Printf("Model %s is not fitted for %s\n",
			notFitted.ModelName, notFitted.Method)
	}

	var valErr *scigoErrors.ValueError
	if errors.As(valueErr, &valErr) {
		fmt.Printf("Value error in %s: %s\n", valErr.Op, valErr.Message)
	}

	// Output: Custom error detected
	// Model LinearRegression is not fitted for Predict
	// Value error in StandardScaler: negative values not supported
}

// Example_errorChaining demonstrates practical error chaining in ML operations
func Example_errorChaining() {
	// Simulate a machine learning pipeline error
	simulateMLError := func() error {
		// Simulate data validation error
		dataErr := fmt.Errorf("invalid data format")

		// Wrap with preprocessing context
		prepErr := fmt.Errorf("data preprocessing failed: %w", dataErr)

		// Wrap with model training context
		trainErr := fmt.Errorf("model training failed: %w", prepErr)

		return trainErr
	}

	err := simulateMLError()

	// Print the full error chain
	fmt.Printf("Error: %v\n", err)

	// Walk through the error chain
	current := err
	level := 0
	for current != nil {
		fmt.Printf("Level %d: %v\n", level, current)
		current = errors.Unwrap(current)
		level++
	}

	// Output: Error: model training failed: data preprocessing failed: invalid data format
	// Level 0: model training failed: data preprocessing failed: invalid data format
	// Level 1: data preprocessing failed: invalid data format
	// Level 2: invalid data format
}

// Example_errorLogging demonstrates structured error logging
func Example_errorLogging() {
	// Create a complex error with context
	baseErr := scigoErrors.NewModelError("SGD", "convergence failure",
		scigoErrors.ErrNotImplemented)

	// Wrap with operation context
	opErr := fmt.Errorf("online learning iteration 150: %w", baseErr)

	// Would log different levels of detail in production
	// slog.Error("Simple error", "error", opErr)
	// slog.Error("Detailed error", "error", fmt.Sprintf("%+v", opErr)) // Stack trace with cockroachdb/errors

	// For production, you'd use structured logging
	fmt.Printf("Error occurred in online learning: %v\n", opErr)

	// Output: Error occurred in online learning: online learning iteration 150: goml: SGD: convergence failure: not implemented
}
