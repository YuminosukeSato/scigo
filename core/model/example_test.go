package model_test

import (
	"fmt"

	"github.com/YuminosukeSato/scigo/core/model"
)

// ExampleBaseEstimator demonstrates BaseEstimator state management
func ExampleBaseEstimator() {
	// Create a BaseEstimator (typically embedded in actual models)
	estimator := &model.BaseEstimator{}

	// Check initial state
	fmt.Printf("Initially fitted: %t\n", estimator.IsFitted())

	// Mark as fitted
	estimator.SetFitted()
	fmt.Printf("After SetFitted: %t\n", estimator.IsFitted())

	// Reset to unfitted state
	estimator.Reset()
	fmt.Printf("After Reset: %t\n", estimator.IsFitted())

	// Output: Initially fitted: false
	// After SetFitted: true
	// After Reset: false
}

// ExampleBaseEstimator_workflowPattern demonstrates typical usage pattern
func ExampleBaseEstimator_workflowPattern() {
	// This example shows how BaseEstimator is typically used in models
	type MyModel struct {
		model.BaseEstimator
		// model-specific fields would go here
	}

	myModel := &MyModel{}

	// Check if model needs training
	if !myModel.IsFitted() {
		fmt.Println("Model needs training")
		
		// Simulate training process
		// ... training logic would go here ...
		
		// Mark as fitted after successful training
		myModel.SetFitted()
		fmt.Println("Model trained successfully")
	}

	// Now model is ready for use
	if myModel.IsFitted() {
		fmt.Println("Model is ready for predictions")
	}

	// Output: Model needs training
	// Model trained successfully
	// Model is ready for predictions
}