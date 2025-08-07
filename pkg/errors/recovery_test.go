package errors

import (
	"errors"
	"fmt"
	"strings"
	"testing"
)

// TestRecover_WithPanic tests the Recover function when a panic occurs
func TestRecover_WithPanic(t *testing.T) {
	testFunc := func() (err error) {
		defer Recover(&err, "TestOperation")
		panic("test panic message")
	}

	err := testFunc()

	if err == nil {
		t.Fatal("Expected error from recovered panic, got nil")
	}

	var panicErr *PanicError
	if !errors.As(err, &panicErr) {
		t.Fatalf("Expected PanicError, got %T", err)
	}

	if panicErr.Operation != "TestOperation" {
		t.Errorf("Expected operation 'TestOperation', got '%s'", panicErr.Operation)
	}

	if panicErr.PanicValue != "test panic message" {
		t.Errorf("Expected panic value 'test panic message', got '%v'", panicErr.PanicValue)
	}

	if panicErr.StackTrace == "" {
		t.Error("Expected non-empty stack trace")
	}

	// Check error message format
	expectedMsg := "panic in TestOperation: test panic message"
	if panicErr.Error() != expectedMsg {
		t.Errorf("Expected error message '%s', got '%s'", expectedMsg, panicErr.Error())
	}
}

// TestRecover_WithoutPanic tests the Recover function when no panic occurs
func TestRecover_WithoutPanic(t *testing.T) {
	testFunc := func() (err error) {
		defer Recover(&err, "TestOperation")
		return nil // Normal return, no panic
	}

	err := testFunc()

	if err != nil {
		t.Fatalf("Expected no error when no panic occurs, got: %v", err)
	}
}

// TestRecover_WithExistingError tests Recover when function has existing error and panic occurs
func TestRecover_WithExistingError(t *testing.T) {
	originalErr := fmt.Errorf("original error")

	testFunc := func() (err error) {
		defer Recover(&err, "TestOperation")
		err = originalErr // Set an error first
		panic("panic after error")
	}

	err := testFunc()

	if err == nil {
		t.Fatal("Expected error from recovered panic with existing error, got nil")
	}

	// Should be a wrapped error containing both panic and original error info
	errMsg := err.Error()
	if !strings.Contains(errMsg, "panic in TestOperation") {
		t.Errorf("Error message should contain panic info: %s", errMsg)
	}

	if !strings.Contains(errMsg, "original error") {
		t.Errorf("Error message should contain original error: %s", errMsg)
	}

	// Should be able to unwrap to original error
	if !errors.Is(err, originalErr) {
		t.Error("Should be able to identify original error with errors.Is")
	}
}

// TestSafeExecute_Success tests SafeExecute with successful function
func TestSafeExecute_Success(t *testing.T) {
	err := SafeExecute("test operation", func() error {
		return nil // Success case
	})

	if err != nil {
		t.Fatalf("Expected no error for successful operation, got: %v", err)
	}
}

// TestSafeExecute_FunctionError tests SafeExecute with function returning error
func TestSafeExecute_FunctionError(t *testing.T) {
	originalErr := fmt.Errorf("function error")

	err := SafeExecute("test operation", func() error {
		return originalErr
	})

	if err != originalErr {
		t.Fatalf("Expected original error, got: %v", err)
	}
}

// TestSafeExecute_Panic tests SafeExecute with panicking function
func TestSafeExecute_Panic(t *testing.T) {
	err := SafeExecute("test operation", func() error {
		panic("test panic in safe execute")
	})

	if err == nil {
		t.Fatal("Expected error from panic in SafeExecute, got nil")
	}

	var panicErr *PanicError
	if !errors.As(err, &panicErr) {
		t.Fatalf("Expected PanicError, got %T", err)
	}

	if panicErr.PanicValue != "test panic in safe execute" {
		t.Errorf("Expected panic value 'test panic in safe execute', got '%v'", panicErr.PanicValue)
	}
}

// TestPanicError_Interface tests PanicError implements error interface properly
func TestPanicError_Interface(t *testing.T) {
	panicErr := NewPanicError("TestOp", "test value")

	// Test Error() method
	expectedMsg := "panic in TestOp: test value"
	if panicErr.Error() != expectedMsg {
		t.Errorf("Expected error message '%s', got '%s'", expectedMsg, panicErr.Error())
	}

	// Test String() method includes stack trace
	str := panicErr.String()
	if !strings.Contains(str, "Stack trace:") {
		t.Error("String() should include stack trace information")
	}

	if !strings.Contains(str, "panic in TestOp: test value") {
		t.Error("String() should include basic error information")
	}

	// Test Unwrap() returns nil
	if panicErr.Unwrap() != nil {
		t.Error("PanicError.Unwrap() should return nil")
	}
}

// TestRecover_DifferentPanicTypes tests Recover with different types of panic values
func TestRecover_DifferentPanicTypes(t *testing.T) {
	testCases := []struct {
		name       string
		panicValue interface{}
		// expectedValue is what we expect to receive (Go converts panic(nil) to a specific string)
		expectedValue interface{}
	}{
		{"string panic", "string panic", "string panic"},
		{"int panic", 42, 42},
		{"error panic", fmt.Errorf("error as panic"), fmt.Errorf("error as panic")},
		{"nil panic", nil, "panic called with nil argument"},
		{"struct panic", struct{ Msg string }{"struct message"}, struct{ Msg string }{"struct message"}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			testFunc := func() (err error) {
				defer Recover(&err, "TypeTest")
				panic(tc.panicValue)
			}

			err := testFunc()

			if err == nil {
				t.Fatal("Expected error from panic")
			}

			var panicErr *PanicError
			if !errors.As(err, &panicErr) {
				t.Fatalf("Expected PanicError, got %T", err)
			}

			if fmt.Sprintf("%v", panicErr.PanicValue) != fmt.Sprintf("%v", tc.expectedValue) {
				t.Errorf("Expected panic value %v, got %v", tc.expectedValue, panicErr.PanicValue)
			}
		})
	}
}

// BenchmarkRecover tests performance overhead of Recover when no panic occurs
func BenchmarkRecover_NoPanic(b *testing.B) {
	for i := 0; i < b.N; i++ {
		func() (err error) {
			defer Recover(&err, "BenchmarkOp")
			// Normal operation, no panic
			return nil
		}()
	}
}

// BenchmarkSafeExecute_NoPanic benchmarks SafeExecute with no panic
func BenchmarkSafeExecute_NoPanic(b *testing.B) {
	for i := 0; i < b.N; i++ {
		SafeExecute("BenchmarkOp", func() error {
			// Normal operation, no panic
			return nil
		})
	}
}
