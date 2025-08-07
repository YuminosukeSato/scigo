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
