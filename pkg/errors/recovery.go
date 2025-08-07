// Package errors provides comprehensive error handling utilities for SciGo.
//
// This file contains panic recovery utilities that help maintain library stability
// by converting unexpected panics into structured errors with debugging information.

package errors

import (
	"fmt"
	"runtime/debug"
)

// PanicError represents an error that was created from a recovered panic.
// It includes the original panic value and stack trace information.
type PanicError struct {
	// PanicValue is the original value passed to panic()
	PanicValue interface{}

	// StackTrace contains the stack trace at the time of panic
	StackTrace string

	// Operation identifies where the panic was recovered
	Operation string
}

// Error implements the error interface for PanicError.
func (e *PanicError) Error() string {
	return fmt.Sprintf("panic in %s: %v", e.Operation, e.PanicValue)
}

// Unwrap returns nil as PanicError doesn't wrap another error by default.
func (e *PanicError) Unwrap() error {
	return nil
}

// String provides detailed information including stack trace.
func (e *PanicError) String() string {
	return fmt.Sprintf("panic in %s: %v\nStack trace:\n%s",
		e.Operation, e.PanicValue, e.StackTrace)
}

// NewPanicError creates a new PanicError with the given operation context and panic value.
func NewPanicError(operation string, panicValue interface{}) *PanicError {
	return &PanicError{
		PanicValue: panicValue,
		StackTrace: string(debug.Stack()),
		Operation:  operation,
	}
}

// Recover is a utility function to be used with defer to recover from panics
// and convert them into errors. It includes stack trace information for debugging.
//
// This function should be called with a pointer to the error return value
// of the function where it's used.
//
// Usage:
//
//	func SomeMethod() (err error) {
//	    defer Recover(&err, "SomeMethod")
//	    // ... method implementation ...
//	    return nil
//	}
//
// If a panic occurs, it will be converted to a PanicError and assigned to err.
// If the function already has an error, the panic information will be wrapped.
func Recover(err *error, operation string) {
	if r := recover(); r != nil {
		panicErr := NewPanicError(operation, r)

		if *err != nil {
			// Wrap existing error with panic information
			*err = fmt.Errorf("panic in %s: %v (original error: %w)",
				operation, r, *err)
		} else {
			// No existing error, return the panic as error
			*err = panicErr
		}
	}
}

// SafeExecute executes a function and recovers from any panic, converting it to an error.
// This is useful for wrapping dangerous operations that might panic.
//
// Parameters:
//   - operation: A descriptive name for the operation being performed
//   - fn: The function to execute safely
//
// Returns:
//   - error: nil if successful, PanicError if panic occurred, or original error from fn
//
// Example:
//
//	err := SafeExecute("matrix inversion", func() error {
//	    // ... potentially panicking code ...
//	    return someOperation()
//	})
func SafeExecute(operation string, fn func() error) (err error) {
	defer Recover(&err, operation)
	return fn()
}
