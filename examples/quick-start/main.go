// Package main demonstrates the quick start example for SciGo
// This example can be run with: go run github.com/YuminosukeSato/scigo/examples/quick-start@latest
package main

import (
	"fmt"
	"log"

	"github.com/YuminosukeSato/scigo/linear"
	"github.com/YuminosukeSato/scigo/preprocessing"
	"gonum.org/v1/gonum/mat"
)

func main() {
	fmt.Println("🚀 SciGo Quick Start Demo")
	fmt.Println("===========================")

	// Create sample data (non-linear relationship for better demo)
	X := mat.NewDense(6, 2, []float64{
		1.0, 2.5,
		2.0, 1.8,
		3.0, 4.2,
		4.0, 3.1,
		5.0, 6.0,
		6.0, 4.9,
	})
	y := mat.NewDense(6, 1, []float64{3.2, 4.1, 7.8, 6.5, 11.2, 9.8})

	fmt.Printf("📊 Training Data: %dx%d matrix\n", X.RawMatrix().Rows, X.RawMatrix().Cols)
	fmt.Printf("🎯 Target Values: %dx%d matrix\n", y.RawMatrix().Rows, y.RawMatrix().Cols)

	// 1. Data Preprocessing
	fmt.Println("\n📋 Step 1: Data Preprocessing")
	scaler := preprocessing.NewStandardScaler(true, true) // withMean=true, withStd=true
	if err := scaler.Fit(X); err != nil {
		log.Fatalf("Failed to fit scaler: %v", err)
	}

	XScaled, err := scaler.Transform(X)
	if err != nil {
		log.Fatalf("Failed to transform data: %v", err)
	}
	fmt.Printf("✅ Data standardized with mean=0 and std=1\n")

	// 2. Model Training
	fmt.Println("\n🧠 Step 2: Model Training")
	model := linear.NewLinearRegression()
	if err := model.Fit(XScaled, y); err != nil {
		log.Fatalf("Failed to fit model: %v", err)
	}
	fmt.Printf("✅ Linear Regression model trained\n")

	// 3. Make Predictions
	fmt.Println("\n🔮 Step 3: Making Predictions")
	predictions, err := model.Predict(XScaled)
	if err != nil {
		log.Fatalf("Failed to make predictions: %v", err)
	}

	fmt.Printf("Predictions vs Actual:\n")
	rows, _ := predictions.Dims()
	for i := 0; i < rows; i++ {
		pred := predictions.At(i, 0)
		actual := y.At(i, 0)
		fmt.Printf("  Sample %d: Predicted=%.2f, Actual=%.2f\n", i+1, pred, actual)
	}

	// 4. Model Evaluation
	fmt.Println("\n📈 Step 4: Model Evaluation")
	score, err := model.Score(XScaled, y)
	if err != nil {
		log.Fatalf("Failed to calculate score: %v", err)
	}
	fmt.Printf("✅ R² Score: %.4f\n", score)

	if score > 0.9 {
		fmt.Println("🎉 Excellent model performance!")
	} else if score > 0.7 {
		fmt.Println("👍 Good model performance!")
	} else {
		fmt.Println("🤔 Model might need improvement")
	}

	fmt.Println("\n🚀 Ready, Set, SciGo!")
	fmt.Println("Learn more at: https://pkg.go.dev/github.com/YuminosukeSato/scigo")
}