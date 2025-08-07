package errors

import (
	"errors"
	"testing"
)

// mockPanicFunction is a helper function that panics with a given value
func mockPanicFunction(panicValue interface{}) func() error {
	return func() error {
		panic(panicValue)
	}
}

// TestPanicRecoveryIntegration tests the complete panic recovery flow
// from a simulated ML operation that panics
func TestPanicRecoveryIntegration(t *testing.T) {
	testCases := []struct {
		name          string
		panicValue    interface{}
		expectedInErr string
		shouldContain []string
	}{
		{
			name:          "String panic recovery",
			panicValue:    "unexpected nil pointer",
			expectedInErr: "panic in MLOperation: unexpected nil pointer",
			shouldContain: []string{"panic in MLOperation", "unexpected nil pointer"},
		},
		{
			name:          "Error panic recovery",
			panicValue:    errors.New("matrix dimension error"),
			expectedInErr: "panic in MLOperation: matrix dimension error",
			shouldContain: []string{"panic in MLOperation", "matrix dimension error"},
		},
		{
			name:          "Integer panic recovery",
			panicValue:    42,
			expectedInErr: "panic in MLOperation: 42",
			shouldContain: []string{"panic in MLOperation", "42"},
		},
		{
			name:          "Nil panic recovery",
			panicValue:    nil,
			expectedInErr: "panic in MLOperation: panic called with nil argument",
			shouldContain: []string{"panic in MLOperation", "panic called with nil argument"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Simulate a machine learning operation that panics
			err := SafeExecute("MLOperation", mockPanicFunction(tc.panicValue))

			if err == nil {
				t.Fatal("Expected error from panic recovery, got nil")
			}

			// Check that we got a PanicError
			var panicErr *PanicError
			if !errors.As(err, &panicErr) {
				t.Fatalf("Expected PanicError, got %T: %v", err, err)
			}

			// Check error message contains expected content
			errMsg := err.Error()
			if errMsg != tc.expectedInErr {
				t.Errorf("Expected error message '%s', got '%s'", tc.expectedInErr, errMsg)
			}

			// Check that all expected strings are present
			for _, expected := range tc.shouldContain {
				if !contains(errMsg, expected) {
					t.Errorf("Error message should contain '%s': %s", expected, errMsg)
				}
			}

			// Check that stack trace is present
			if panicErr.StackTrace == "" {
				t.Error("Expected non-empty stack trace")
			}

			// Check operation context
			if panicErr.Operation != "MLOperation" {
				t.Errorf("Expected operation 'MLOperation', got '%s'", panicErr.Operation)
			}
		})
	}
}

// TestPanicRecoveryWithDeferRecover tests the defer-based recovery pattern
func TestPanicRecoveryWithDeferRecover(t *testing.T) {
	simulateMLFunction := func() (err error) {
		defer Recover(&err, "SimulatedML.Fit")

		// Simulate some successful operations first
		_ = "preprocessing complete"

		// Then panic occurs
		panic("matrix inversion failed")
	}

	err := simulateMLFunction()

	if err == nil {
		t.Fatal("Expected error from panic recovery, got nil")
	}

	var panicErr *PanicError
	if !errors.As(err, &panicErr) {
		t.Fatalf("Expected PanicError, got %T", err)
	}

	expectedMsg := "panic in SimulatedML.Fit: matrix inversion failed"
	if panicErr.Error() != expectedMsg {
		t.Errorf("Expected error message '%s', got '%s'", expectedMsg, panicErr.Error())
	}
}

// TestPanicRecoveryWithExistingError tests panic recovery when function already has an error
func TestPanicRecoveryWithExistingError(t *testing.T) {
	originalErr := errors.New("validation failed")

	simulateMLFunction := func() (err error) {
		defer Recover(&err, "MLFunction")

		// Set an error first
		err = originalErr

		// Then panic occurs
		panic("unexpected panic after error")
	}

	err := simulateMLFunction()

	if err == nil {
		t.Fatal("Expected error from panic recovery with existing error, got nil")
	}

	// Should contain both panic info and be traceable to original error
	errMsg := err.Error()
	expectedContains := []string{
		"panic in MLFunction",
		"unexpected panic after error",
		"original error",
		"validation failed",
	}

	for _, expected := range expectedContains {
		if !contains(errMsg, expected) {
			t.Errorf("Error message should contain '%s': %s", expected, errMsg)
		}
	}

	// Should be able to unwrap to original error
	if !errors.Is(err, originalErr) {
		t.Error("Should be able to identify original error with errors.Is")
	}
}

// TestPanicRecoveryChaining tests chaining multiple operations with panic recovery
func TestPanicRecoveryChaining(t *testing.T) {
	// Simulate a chain: preprocessing -> training -> prediction
	preprocessing := func() error {
		return SafeExecute("Preprocessing", func() error {
			return nil // Success
		})
	}

	training := func() error {
		return SafeExecute("Training", func() error {
			panic("convergence failure")
		})
	}

	prediction := func() error {
		return SafeExecute("Prediction", func() error {
			return nil // This won't be reached due to training panic
		})
	}

	// Run the pipeline
	if err := preprocessing(); err != nil {
		t.Fatalf("Preprocessing should not fail: %v", err)
	}

	err := training()
	if err == nil {
		t.Fatal("Training should fail due to panic")
	}

	var panicErr *PanicError
	if !errors.As(err, &panicErr) {
		t.Fatalf("Expected PanicError from training, got %T", err)
	}

	if panicErr.Operation != "Training" {
		t.Errorf("Expected operation 'Training', got '%s'", panicErr.Operation)
	}

	// Prediction should still work if called independently
	if err := prediction(); err != nil {
		t.Fatalf("Prediction should not fail: %v", err)
	}
}

// TestNoPanicScenario tests that normal operations are not affected by panic recovery
func TestNoPanicScenario(t *testing.T) {
	normalOperation := func() (err error) {
		defer Recover(&err, "NormalOperation")

		// Normal operations without panic
		result := 2 + 2
		if result != 4 {
			return errors.New("math is broken")
		}

		return nil
	}

	err := normalOperation()
	if err != nil {
		t.Fatalf("Normal operation should not produce error: %v", err)
	}
}

// TestPanicRecoveryPerformance benchmarks the performance overhead
func BenchmarkPanicRecoveryOverhead(b *testing.B) {
	b.Run("WithRecover", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			func() (err error) {
				defer Recover(&err, "BenchOperation")
				// Minimal work
				_ = i * 2
				return nil
			}()
		}
	})

	b.Run("WithoutRecover", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			func() error {
				// Same minimal work, no recovery
				_ = i * 2
				return nil
			}()
		}
	})
}

// contains is a helper function to check if a string contains a substring
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 ||
		func() bool {
			for i := 0; i <= len(s)-len(substr); i++ {
				if s[i:i+len(substr)] == substr {
					return true
				}
			}
			return false
		}())
}
