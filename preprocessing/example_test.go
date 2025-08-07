package preprocessing_test

import (
	"fmt"
	"log"

	"github.com/YuminosukeSato/scigo/preprocessing"
	"gonum.org/v1/gonum/mat"
)

// ExampleStandardScaler demonstrates basic usage of StandardScaler
func ExampleStandardScaler() {
	// Create sample training data
	data := []float64{
		1.0, 2.0,
		3.0, 4.0,
		5.0, 6.0,
		7.0, 8.0,
	}
	X := mat.NewDense(4, 2, data)

	// Create and fit scaler
	scaler := preprocessing.NewStandardScaler(true, true)
	err := scaler.Fit(X)
	if err != nil {
		log.Fatal(err)
	}

	// Transform the data
	scaled, err := scaler.Transform(X)
	if err != nil {
		log.Fatal(err)
	}

	// Print first row of scaled data
	fmt.Printf("Scaled first row: [%.2f, %.2f]\n", scaled.At(0, 0), scaled.At(0, 1))

	// Output: Scaled first row: [-1.34, -1.34]
}

// ExampleStandardScaler_fitTransform demonstrates FitTransform usage
func ExampleStandardScaler_fitTransform() {
	// Create sample data
	data := []float64{
		10.0, 100.0,
		20.0, 200.0,
		30.0, 300.0,
	}
	X := mat.NewDense(3, 2, data)

	// Create scaler and fit+transform in one step
	scaler := preprocessing.NewStandardScaler(true, true)
	scaled, err := scaler.FitTransform(X)
	if err != nil {
		log.Fatal(err)
	}

	// Check that scaler is now fitted
	if scaler.IsFitted() {
		fmt.Println("Scaler is fitted")
	}

	// Print dimensions
	r, c := scaled.Dims()
	fmt.Printf("Scaled data shape: (%d, %d)\n", r, c)

	// Output: Scaler is fitted
	// Scaled data shape: (3, 2)
}

// ExampleMinMaxScaler demonstrates basic MinMaxScaler usage
func ExampleMinMaxScaler() {
	// Create sample data
	data := []float64{
		1.0, 10.0,
		2.0, 20.0,
		3.0, 30.0,
		4.0, 40.0,
	}
	X := mat.NewDense(4, 2, data)

	// Create MinMaxScaler for [0, 1] range
	scaler := preprocessing.NewMinMaxScaler([2]float64{0.0, 1.0})

	// Fit and transform
	scaled, err := scaler.FitTransform(X)
	if err != nil {
		log.Fatal(err)
	}

	// Print first and last values (should be 0.0 and 1.0)
	fmt.Printf("First row: [%.1f, %.1f]\n", scaled.At(0, 0), scaled.At(0, 1))
	fmt.Printf("Last row: [%.1f, %.1f]\n", scaled.At(3, 0), scaled.At(3, 1))

	// Output: First row: [0.0, 0.0]
	// Last row: [1.0, 1.0]
}

// ExampleMinMaxScaler_customRange demonstrates custom range scaling
func ExampleMinMaxScaler_customRange() {
	// Create sample data
	data := []float64{
		0.0,
		5.0,
		10.0,
	}
	X := mat.NewDense(3, 1, data)

	// Create MinMaxScaler for [-1, 1] range
	scaler := preprocessing.NewMinMaxScaler([2]float64{-1.0, 1.0})
	scaled, err := scaler.FitTransform(X)
	if err != nil {
		log.Fatal(err)
	}

	// Print scaled values
	for i := 0; i < 3; i++ {
		fmt.Printf("%.1f -> %.1f\n", X.At(i, 0), scaled.At(i, 0))
	}

	// Output: 0.0 -> -1.0
	// 5.0 -> 0.0
	// 10.0 -> 1.0
}

// ExampleOneHotEncoder demonstrates OneHotEncoder usage
func ExampleOneHotEncoder() {
	// Create sample categorical data
	data := [][]string{
		{"red"},
		{"green"},
		{"blue"},
		{"red"},
	}

	// Create and fit encoder
	encoder := preprocessing.NewOneHotEncoder()
	err := encoder.Fit(data)
	if err != nil {
		log.Fatal(err)
	}

	// Transform the data
	encoded, err := encoder.Transform(data)
	if err != nil {
		log.Fatal(err)
	}

	// Print feature names
	features := encoder.GetFeatureNamesOut(nil)
	fmt.Printf("Features: %v\n", features)

	// Print encoded shape
	r, c := encoded.Dims()
	fmt.Printf("Encoded shape: (%d, %d)\n", r, c)

	// Output: Features: [x0_blue x0_green x0_red]
	// Encoded shape: (4, 3)
}

// ExampleStandardScaler_inverseTransform demonstrates inverse transformation
func ExampleStandardScaler_inverseTransform() {
	// Original data
	data := []float64{
		2.0, 4.0,
		6.0, 8.0,
	}
	X := mat.NewDense(2, 2, data)

	// Standardize
	scaler := preprocessing.NewStandardScaler(true, true)
	scaled, err := scaler.FitTransform(X)
	if err != nil {
		log.Fatal(err)
	}

	// Inverse transform back to original scale
	restored, err := scaler.InverseTransform(scaled)
	if err != nil {
		log.Fatal(err)
	}

	// Check if values match original (within floating point precision)
	fmt.Printf("Original: [%.1f, %.1f]\n", X.At(0, 0), X.At(0, 1))
	fmt.Printf("Restored: [%.1f, %.1f]\n", restored.At(0, 0), restored.At(0, 1))

	// Output: Original: [2.0, 4.0]
	// Restored: [2.0, 4.0]
}
