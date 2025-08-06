package errors_test

import (
	"errors"
	"fmt"
	"testing"

	scigoErrors "github.com/YuminosukeSato/scigo/pkg/errors"
)

// TestErrorWrappingCompatibility tests Go 1.13+ error wrapping with our custom types
func TestErrorWrappingCompatibility(t *testing.T) {
	// Create a custom error
	originalErr := scigoErrors.NewNotFittedError("TestModel", "Predict")
	
	// Wrap it with Go 1.13+ syntax
	wrappedErr := fmt.Errorf("pipeline step failed: %w", originalErr)
	
	// Test errors.Is functionality
	if !errors.Is(wrappedErr, originalErr) {
		t.Errorf("errors.Is failed to identify wrapped error")
	}
	
	// Test errors.As functionality
	var notFittedErr *scigoErrors.NotFittedError
	if !errors.As(wrappedErr, &notFittedErr) {
		t.Errorf("errors.As failed to extract NotFittedError")
	}
	
	if notFittedErr.ModelName != "TestModel" {
		t.Errorf("expected ModelName 'TestModel', got '%s'", notFittedErr.ModelName)
	}
}

// TestErrorChainTraversal tests error chain traversal
func TestErrorChainTraversal(t *testing.T) {
	// Create a chain of errors
	level3 := fmt.Errorf("database connection failed")
	level2 := fmt.Errorf("data loading failed: %w", level3)
	level1 := fmt.Errorf("model training failed: %w", level2)
	
	// Test unwrapping
	unwrapped1 := errors.Unwrap(level1)
	if unwrapped1.Error() != level2.Error() {
		t.Errorf("first unwrap failed")
	}
	
	unwrapped2 := errors.Unwrap(unwrapped1)
	if unwrapped2.Error() != level3.Error() {
		t.Errorf("second unwrap failed")
	}
	
	// Test that we can find the root cause
	if !errors.Is(level1, level3) {
		t.Errorf("errors.Is failed to find root cause")
	}
}

// TestCombinedErrorTypes tests mixing custom and standard errors
func TestCombinedErrorTypes(t *testing.T) {
	// Standard error
	stdErr := fmt.Errorf("standard error")
	
	// Custom error wrapping standard error
	customErr := scigoErrors.NewModelError("TestOp", "test failure", stdErr)
	
	// Wrap custom error with Go 1.13+ syntax
	wrappedErr := fmt.Errorf("operation context: %w", customErr)
	
	// Test that we can find both types
	if !errors.Is(wrappedErr, stdErr) {
		t.Errorf("failed to find standard error in chain")
	}
	
	var modelErr *scigoErrors.ModelError
	if !errors.As(wrappedErr, &modelErr) {
		t.Errorf("failed to extract ModelError")
	}
	
	// Test that ModelError's Unwrap method works
	if modelErr.Unwrap() != stdErr {
		t.Errorf("ModelError.Unwrap() didn't return expected error")
	}
}

// TestSentinelErrors tests sentinel error patterns
func TestSentinelErrors(t *testing.T) {
	// Test with our predefined sentinel errors
	err := scigoErrors.NewModelError("TestOp", "empty data", scigoErrors.ErrEmptyData)
	
	if !errors.Is(err, scigoErrors.ErrEmptyData) {
		t.Errorf("failed to identify ErrEmptyData sentinel")
	}
	
	// Wrap and test again
	wrappedErr := fmt.Errorf("preprocessing failed: %w", err)
	
	if !errors.Is(wrappedErr, scigoErrors.ErrEmptyData) {
		t.Errorf("failed to identify ErrEmptyData through wrapper")
	}
}