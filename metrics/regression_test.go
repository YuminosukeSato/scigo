package metrics

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestMSE(t *testing.T) {
	tests := []struct {
		name      string
		yTrue     *mat.VecDense
		yPred     *mat.VecDense
		want      float64
		tolerance float64
		wantErr   bool
	}{
		{
			name:      "perfect prediction",
			yTrue:     mat.NewVecDense(5, []float64{1.0, 2.0, 3.0, 4.0, 5.0}),
			yPred:     mat.NewVecDense(5, []float64{1.0, 2.0, 3.0, 4.0, 5.0}),
			want:      0.0,
			tolerance: 1e-10,
			wantErr:   false,
		},
		{
			name:      "simple case",
			yTrue:     mat.NewVecDense(4, []float64{1.0, 2.0, 3.0, 4.0}),
			yPred:     mat.NewVecDense(4, []float64{1.5, 2.5, 2.5, 3.5}),
			want:      0.25, // ((0.5)^2 + (0.5)^2 + (-0.5)^2 + (-0.5)^2) / 4 = 1.0/4 = 0.25
			tolerance: 1e-10,
			wantErr:   false,
		},
		{
			name:      "larger errors",
			yTrue:     mat.NewVecDense(3, []float64{10.0, 20.0, 30.0}),
			yPred:     mat.NewVecDense(3, []float64{12.0, 18.0, 33.0}),
			want:      17.0 / 3.0, // ((2)^2 + (-2)^2 + (3)^2) / 3 = (4 + 4 + 9) / 3 = 17/3 ≈ 5.67
			tolerance: 1e-10,
			wantErr:   false,
		},
		{
			name:      "dimension mismatch",
			yTrue:     mat.NewVecDense(3, []float64{1.0, 2.0, 3.0}),
			yPred:     mat.NewVecDense(2, []float64{1.0, 2.0}),
			want:      0.0,
			tolerance: 1e-10,
			wantErr:   true,
		},
		{
			name:      "empty vectors",
			yTrue:     &mat.VecDense{},
			yPred:     &mat.VecDense{},
			want:      0.0,
			tolerance: 1e-10,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := MSE(tt.yTrue, tt.yPred)

			if (err != nil) != tt.wantErr {
				t.Errorf("MSE() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if math.Abs(got-tt.want) > tt.tolerance {
					t.Errorf("MSE() = %v, want %v (tolerance: %v)", got, tt.want, tt.tolerance)
				}
			}
		})
	}
}

func TestMSEWithMatrices(t *testing.T) {
	tests := []struct {
		name      string
		yTrue     mat.Matrix
		yPred     mat.Matrix
		want      float64
		tolerance float64
		wantErr   bool
	}{
		{
			name: "matrix input - single column",
			yTrue: mat.NewDense(4, 1, []float64{
				1.0,
				2.0,
				3.0,
				4.0,
			}),
			yPred: mat.NewDense(4, 1, []float64{
				1.5,
				2.5,
				2.5,
				3.5,
			}),
			want:      0.25,
			tolerance: 1e-10,
			wantErr:   false,
		},
		{
			name: "matrix input - multiple columns should error",
			yTrue: mat.NewDense(2, 2, []float64{
				1.0, 2.0,
				3.0, 4.0,
			}),
			yPred: mat.NewDense(2, 2, []float64{
				1.0, 2.0,
				3.0, 4.0,
			}),
			want:    0.0,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := MSEMatrix(tt.yTrue, tt.yPred)

			if (err != nil) != tt.wantErr {
				t.Errorf("MSEMatrix() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if math.Abs(got-tt.want) > tt.tolerance {
					t.Errorf("MSEMatrix() = %v, want %v", got, tt.want)
				}
			}
		})
	}
}

func TestRMSE(t *testing.T) {
	tests := []struct {
		name      string
		yTrue     *mat.VecDense
		yPred     *mat.VecDense
		want      float64
		tolerance float64
		wantErr   bool
	}{
		{
			name:      "perfect prediction",
			yTrue:     mat.NewVecDense(5, []float64{1.0, 2.0, 3.0, 4.0, 5.0}),
			yPred:     mat.NewVecDense(5, []float64{1.0, 2.0, 3.0, 4.0, 5.0}),
			want:      0.0,
			tolerance: 1e-10,
			wantErr:   false,
		},
		{
			name:      "simple case",
			yTrue:     mat.NewVecDense(4, []float64{0.0, 0.0, 0.0, 0.0}),
			yPred:     mat.NewVecDense(4, []float64{1.0, 1.0, 1.0, 1.0}),
			want:      1.0, // sqrt(MSE) = sqrt(1.0) = 1.0
			tolerance: 1e-10,
			wantErr:   false,
		},
		{
			name:      "dimension mismatch",
			yTrue:     mat.NewVecDense(3, []float64{1.0, 2.0, 3.0}),
			yPred:     mat.NewVecDense(2, []float64{1.0, 2.0}),
			want:      0.0,
			tolerance: 1e-10,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := RMSE(tt.yTrue, tt.yPred)

			if (err != nil) != tt.wantErr {
				t.Errorf("RMSE() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if math.Abs(got-tt.want) > tt.tolerance {
					t.Errorf("RMSE() = %v, want %v", got, tt.want)
				}
			}
		})
	}
}

func TestMAE(t *testing.T) {
	tests := []struct {
		name      string
		yTrue     *mat.VecDense
		yPred     *mat.VecDense
		want      float64
		tolerance float64
		wantErr   bool
	}{
		{
			name:      "perfect prediction",
			yTrue:     mat.NewVecDense(5, []float64{1.0, 2.0, 3.0, 4.0, 5.0}),
			yPred:     mat.NewVecDense(5, []float64{1.0, 2.0, 3.0, 4.0, 5.0}),
			want:      0.0,
			tolerance: 1e-10,
			wantErr:   false,
		},
		{
			name:      "simple case",
			yTrue:     mat.NewVecDense(4, []float64{1.0, 2.0, 3.0, 4.0}),
			yPred:     mat.NewVecDense(4, []float64{1.5, 2.5, 2.5, 3.5}),
			want:      0.5, // (0.5 + 0.5 + 0.5 + 0.5) / 4 = 0.5
			tolerance: 1e-10,
			wantErr:   false,
		},
		{
			name:      "with negative differences",
			yTrue:     mat.NewVecDense(4, []float64{1.0, 2.0, 3.0, 4.0}),
			yPred:     mat.NewVecDense(4, []float64{2.0, 1.0, 4.0, 3.0}),
			want:      1.0, // (1.0 + 1.0 + 1.0 + 1.0) / 4 = 1.0
			tolerance: 1e-10,
			wantErr:   false,
		},
		{
			name:      "dimension mismatch",
			yTrue:     mat.NewVecDense(3, []float64{1.0, 2.0, 3.0}),
			yPred:     mat.NewVecDense(2, []float64{1.0, 2.0}),
			want:      0.0,
			tolerance: 1e-10,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := MAE(tt.yTrue, tt.yPred)

			if (err != nil) != tt.wantErr {
				t.Errorf("MAE() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if math.Abs(got-tt.want) > tt.tolerance {
					t.Errorf("MAE() = %v, want %v", got, tt.want)
				}
			}
		})
	}
}

func TestR2Score(t *testing.T) {
	tests := []struct {
		name      string
		yTrue     *mat.VecDense
		yPred     *mat.VecDense
		want      float64
		tolerance float64
		wantErr   bool
	}{
		{
			name:      "perfect prediction",
			yTrue:     mat.NewVecDense(5, []float64{1.0, 2.0, 3.0, 4.0, 5.0}),
			yPred:     mat.NewVecDense(5, []float64{1.0, 2.0, 3.0, 4.0, 5.0}),
			want:      1.0,
			tolerance: 1e-10,
			wantErr:   false,
		},
		{
			name:      "no variance in yTrue",
			yTrue:     mat.NewVecDense(5, []float64{3.0, 3.0, 3.0, 3.0, 3.0}),
			yPred:     mat.NewVecDense(5, []float64{2.0, 3.0, 4.0, 3.0, 3.0}),
			want:      0.0,
			tolerance: 1e-10,
			wantErr:   true, // Error when total variation is 0
		},
		{
			name:      "worse than mean baseline",
			yTrue:     mat.NewVecDense(4, []float64{1.0, 2.0, 3.0, 4.0}),
			yPred:     mat.NewVecDense(4, []float64{4.0, 3.0, 2.0, 1.0}),
			want:      -3.0, // Negative R² value (worse than mean prediction)
			tolerance: 0.01,
			wantErr:   false,
		},
		{
			name:      "dimension mismatch",
			yTrue:     mat.NewVecDense(3, []float64{1.0, 2.0, 3.0}),
			yPred:     mat.NewVecDense(2, []float64{1.0, 2.0}),
			want:      0.0,
			tolerance: 1e-10,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := R2Score(tt.yTrue, tt.yPred)

			if (err != nil) != tt.wantErr {
				t.Errorf("R2Score() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if math.Abs(got-tt.want) > tt.tolerance {
					t.Errorf("R2Score() = %v, want %v", got, tt.want)
				}
			}
		})
	}
}

func TestMAPE(t *testing.T) {
	tests := []struct {
		name      string
		yTrue     *mat.VecDense
		yPred     *mat.VecDense
		want      float64
		tolerance float64
		wantErr   bool
	}{
		{
			name:      "perfect prediction",
			yTrue:     mat.NewVecDense(5, []float64{1.0, 2.0, 3.0, 4.0, 5.0}),
			yPred:     mat.NewVecDense(5, []float64{1.0, 2.0, 3.0, 4.0, 5.0}),
			want:      0.0,
			tolerance: 1e-10,
			wantErr:   false,
		},
		{
			name:      "simple case",
			yTrue:     mat.NewVecDense(4, []float64{10.0, 20.0, 30.0, 40.0}),
			yPred:     mat.NewVecDense(4, []float64{11.0, 18.0, 33.0, 36.0}),
			want:      10.0, // (|10-11|/|10| + |20-18|/|20| + |30-33|/|30| + |40-36|/|40|) * 100 / 4 = (0.1 + 0.1 + 0.1 + 0.1) * 100 / 4 = 10%
			tolerance: 1e-9,
			wantErr:   false,
		},
		{
			name:      "with zeros should skip zero values",
			yTrue:     mat.NewVecDense(4, []float64{0.0, 2.0, 0.0, 4.0}),
			yPred:     mat.NewVecDense(4, []float64{1.0, 2.2, 1.0, 4.4}),
			want:      10.0, // Only non-zero values: (|2-2.2|/|2| + |4-4.4|/|4|) * 100 / 2 = (0.1 + 0.1) * 100 / 2 = 10%
			tolerance: 1e-9,
			wantErr:   false,
		},
		{
			name:      "all zeros should error",
			yTrue:     mat.NewVecDense(3, []float64{0.0, 0.0, 0.0}),
			yPred:     mat.NewVecDense(3, []float64{1.0, 2.0, 3.0}),
			want:      0.0,
			tolerance: 1e-10,
			wantErr:   true,
		},
		{
			name:      "dimension mismatch",
			yTrue:     mat.NewVecDense(3, []float64{1.0, 2.0, 3.0}),
			yPred:     mat.NewVecDense(2, []float64{1.0, 2.0}),
			want:      0.0,
			tolerance: 1e-10,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := MAPE(tt.yTrue, tt.yPred)

			if (err != nil) != tt.wantErr {
				t.Errorf("MAPE() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if math.Abs(got-tt.want) > tt.tolerance {
					t.Errorf("MAPE() = %v, want %v (tolerance: %v)", got, tt.want, tt.tolerance)
				}
			}
		})
	}
}

func TestExplainedVarianceScore(t *testing.T) {
	tests := []struct {
		name      string
		yTrue     *mat.VecDense
		yPred     *mat.VecDense
		want      float64
		tolerance float64
		wantErr   bool
	}{
		{
			name:      "perfect prediction",
			yTrue:     mat.NewVecDense(5, []float64{1.0, 2.0, 3.0, 4.0, 5.0}),
			yPred:     mat.NewVecDense(5, []float64{1.0, 2.0, 3.0, 4.0, 5.0}),
			want:      1.0,
			tolerance: 1e-10,
			wantErr:   false,
		},
		{
			name:      "systematic offset but correct variance",
			yTrue:     mat.NewVecDense(4, []float64{1.0, 2.0, 3.0, 4.0}),
			yPred:     mat.NewVecDense(4, []float64{2.0, 3.0, 4.0, 5.0}), // +1 offset
			want:      1.0,                                               // Perfect variance explanation despite offset
			tolerance: 1e-10,
			wantErr:   false,
		},
		{
			name:      "no variance in yTrue",
			yTrue:     mat.NewVecDense(5, []float64{3.0, 3.0, 3.0, 3.0, 3.0}),
			yPred:     mat.NewVecDense(5, []float64{2.0, 3.0, 4.0, 3.0, 3.0}),
			want:      0.0,
			tolerance: 1e-10,
			wantErr:   true,
		},
		{
			name:      "poor variance explanation",
			yTrue:     mat.NewVecDense(4, []float64{1.0, 2.0, 3.0, 4.0}),
			yPred:     mat.NewVecDense(4, []float64{2.5, 2.5, 2.5, 2.5}), // Constant prediction
			want:      0.0,                                               // No variance captured
			tolerance: 0.01,
			wantErr:   false,
		},
		{
			name:      "dimension mismatch",
			yTrue:     mat.NewVecDense(3, []float64{1.0, 2.0, 3.0}),
			yPred:     mat.NewVecDense(2, []float64{1.0, 2.0}),
			want:      0.0,
			tolerance: 1e-10,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ExplainedVarianceScore(tt.yTrue, tt.yPred)

			if (err != nil) != tt.wantErr {
				t.Errorf("ExplainedVarianceScore() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if math.Abs(got-tt.want) > tt.tolerance {
					t.Errorf("ExplainedVarianceScore() = %v, want %v (tolerance: %v)", got, tt.want, tt.tolerance)
				}
			}
		})
	}
}

// Edge case tests for numerical stability
func TestNumericalStability(t *testing.T) {
	t.Run("MSE with very large values", func(t *testing.T) {
		yTrue := mat.NewVecDense(3, []float64{1e6, 2e6, 3e6})
		yPred := mat.NewVecDense(3, []float64{1.1e6, 1.9e6, 3.1e6})

		mse, err := MSE(yTrue, yPred)
		if err != nil {
			t.Fatalf("MSE() with large values failed: %v", err)
		}

		expected := ((1e5)*(1e5) + (1e5)*(1e5) + (1e5)*(1e5)) / 3.0 // 1e10/3
		if math.Abs(mse-expected) > 1e-6*expected {
			t.Errorf("MSE() with large values = %v, want %v", mse, expected)
		}
	})

	t.Run("MSE with very small values", func(t *testing.T) {
		yTrue := mat.NewVecDense(3, []float64{1e-6, 2e-6, 3e-6})
		yPred := mat.NewVecDense(3, []float64{1.1e-6, 1.9e-6, 3.1e-6})

		mse, err := MSE(yTrue, yPred)
		if err != nil {
			t.Fatalf("MSE() with small values failed: %v", err)
		}

		if math.IsNaN(mse) || math.IsInf(mse, 0) {
			t.Errorf("MSE() with small values produced NaN or Inf: %v", mse)
		}
	})

	t.Run("R2Score with near-zero variance", func(t *testing.T) {
		epsilon := 1e-15
		yTrue := mat.NewVecDense(3, []float64{1.0, 1.0 + epsilon, 1.0 - epsilon})
		yPred := mat.NewVecDense(3, []float64{1.0, 1.0, 1.0})

		// This should either work or fail gracefully, not crash
		_, err := R2Score(yTrue, yPred)
		// We don't check the exact result because it's numerically unstable
		// Just ensure it doesn't crash
		_ = err
	})
}

// Benchmark tests
func BenchmarkMSE(b *testing.B) {
	size := 10000
	yTrue := mat.NewVecDense(size, nil)
	yPred := mat.NewVecDense(size, nil)

	// Generate random data
	for i := 0; i < size; i++ {
		yTrue.SetVec(i, float64(i))
		yPred.SetVec(i, float64(i)+0.1*float64(i%10))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = MSE(yTrue, yPred)
	}
}

func BenchmarkMAE(b *testing.B) {
	size := 10000
	yTrue := mat.NewVecDense(size, nil)
	yPred := mat.NewVecDense(size, nil)

	for i := 0; i < size; i++ {
		yTrue.SetVec(i, float64(i))
		yPred.SetVec(i, float64(i)+0.1*float64(i%10))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = MAE(yTrue, yPred)
	}
}

func BenchmarkR2Score(b *testing.B) {
	size := 10000
	yTrue := mat.NewVecDense(size, nil)
	yPred := mat.NewVecDense(size, nil)

	for i := 0; i < size; i++ {
		yTrue.SetVec(i, float64(i))
		yPred.SetVec(i, float64(i)+0.1*float64(i%10))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = R2Score(yTrue, yPred)
	}
}
