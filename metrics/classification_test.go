package metrics

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestAUC(t *testing.T) {
	tests := []struct {
		name    string
		yTrue   []float64
		yPred   []float64
		want    float64
		wantErr bool
	}{
		{
			name:  "Perfect classifier",
			yTrue: []float64{0, 0, 0, 1, 1, 1},
			yPred: []float64{0.1, 0.2, 0.3, 0.7, 0.8, 0.9},
			want:  1.0,
		},
		{
			name:  "Worst classifier",
			yTrue: []float64{0, 0, 0, 1, 1, 1},
			yPred: []float64{0.9, 0.8, 0.7, 0.3, 0.2, 0.1},
			want:  0.0,
		},
		{
			name:  "Random classifier",
			yTrue: []float64{0, 1, 0, 1},
			yPred: []float64{0.5, 0.5, 0.5, 0.5},
			want:  0.5,
		},
		{
			name:  "Typical case",
			yTrue: []float64{0, 0, 1, 1},
			yPred: []float64{0.1, 0.4, 0.35, 0.8},
			want:  0.75,
		},
		{
			name:  "All positive labels",
			yTrue: []float64{1, 1, 1, 1},
			yPred: []float64{0.1, 0.4, 0.35, 0.8},
			want:  0.5, // Undefined case, returns 0.5
		},
		{
			name:  "All negative labels",
			yTrue: []float64{0, 0, 0, 0},
			yPred: []float64{0.1, 0.4, 0.35, 0.8},
			want:  0.5, // Undefined case, returns 0.5
		},
		{
			name:    "Non-binary labels",
			yTrue:   []float64{0, 0.5, 1},
			yPred:   []float64{0.1, 0.5, 0.9},
			wantErr: true,
		},
		{
			name:    "Dimension mismatch",
			yTrue:   []float64{0, 1},
			yPred:   []float64{0.5},
			wantErr: true,
		},
		{
			name:    "Empty vectors",
			yTrue:   []float64{},
			yPred:   []float64{},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var yTrue, yPred *mat.VecDense
			if len(tt.yTrue) > 0 {
				yTrue = mat.NewVecDense(len(tt.yTrue), tt.yTrue)
			}
			if len(tt.yPred) > 0 {
				yPred = mat.NewVecDense(len(tt.yPred), tt.yPred)
			}

			got, err := AUC(yTrue, yPred)
			if (err != nil) != tt.wantErr {
				t.Errorf("AUC() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && math.Abs(got-tt.want) > 1e-6 {
				t.Errorf("AUC() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAUCMatrix(t *testing.T) {
	tests := []struct {
		name    string
		yTrue   mat.Matrix
		yPred   mat.Matrix
		want    float64
		wantErr bool
	}{
		{
			name:  "Matrix input",
			yTrue: mat.NewDense(4, 1, []float64{0, 0, 1, 1}),
			yPred: mat.NewDense(4, 1, []float64{0.1, 0.4, 0.35, 0.8}),
			want:  0.75,
		},
		{
			name:  "Multi-column matrix (uses first column)",
			yTrue: mat.NewDense(4, 2, []float64{0, 9, 0, 9, 1, 9, 1, 9}),
			yPred: mat.NewDense(4, 2, []float64{0.1, 9, 0.4, 9, 0.35, 9, 0.8, 9}),
			want:  0.75,
		},
		{
			name:    "Nil matrix",
			yTrue:   nil,
			yPred:   mat.NewDense(1, 1, []float64{0.5}),
			wantErr: true,
		},
		{
			name:    "Empty matrix",
			yTrue:   &mat.Dense{},
			yPred:   &mat.Dense{},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := AUCMatrix(tt.yTrue, tt.yPred)
			if (err != nil) != tt.wantErr {
				t.Errorf("AUCMatrix() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && math.Abs(got-tt.want) > 1e-6 {
				t.Errorf("AUCMatrix() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBinaryLogLoss(t *testing.T) {
	tests := []struct {
		name    string
		yTrue   []float64
		yPred   []float64
		want    float64
		wantErr bool
	}{
		{
			name:  "Perfect predictions",
			yTrue: []float64{0, 0, 1, 1},
			yPred: []float64{0, 0, 1, 1},
			want:  0.0, // Will be small epsilon value due to clipping
		},
		{
			name:  "Typical case",
			yTrue: []float64{0, 0, 1, 1},
			yPred: []float64{0.1, 0.2, 0.8, 0.9},
			want:  0.164252, // Approximate expected value
		},
		{
			name:  "Worst predictions",
			yTrue: []float64{0, 0, 1, 1},
			yPred: []float64{0.9, 0.9, 0.1, 0.1},
			want:  2.3025851, // Approximate expected value
		},
		{
			name:  "Clipping edge case",
			yTrue: []float64{0, 1},
			yPred: []float64{0, 1}, // Will be clipped to avoid log(0)
			want:  0.0,             // Small value due to epsilon
		},
		{
			name:    "Non-binary labels",
			yTrue:   []float64{0, 0.5, 1},
			yPred:   []float64{0.1, 0.5, 0.9},
			wantErr: true,
		},
		{
			name:    "Empty vectors",
			yTrue:   []float64{},
			yPred:   []float64{},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var yTrue, yPred *mat.VecDense
			if len(tt.yTrue) > 0 {
				yTrue = mat.NewVecDense(len(tt.yTrue), tt.yTrue)
			}
			if len(tt.yPred) > 0 {
				yPred = mat.NewVecDense(len(tt.yPred), tt.yPred)
			}

			got, err := BinaryLogLoss(yTrue, yPred)
			if (err != nil) != tt.wantErr {
				t.Errorf("BinaryLogLoss() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && math.Abs(got-tt.want) > 0.01 {
				t.Errorf("BinaryLogLoss() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestClassificationError(t *testing.T) {
	tests := []struct {
		name    string
		yTrue   []float64
		yPred   []float64
		want    float64
		wantErr bool
	}{
		{
			name:  "Perfect classification",
			yTrue: []float64{0, 1, 2, 1, 0},
			yPred: []float64{0, 1, 2, 1, 0},
			want:  0.0,
		},
		{
			name:  "One error",
			yTrue: []float64{0, 1, 2, 1, 0},
			yPred: []float64{0, 1, 1, 1, 0},
			want:  0.2,
		},
		{
			name:  "All wrong",
			yTrue: []float64{0, 0, 0},
			yPred: []float64{1, 1, 1},
			want:  1.0,
		},
		{
			name:  "Binary classification",
			yTrue: []float64{0, 0, 1, 1},
			yPred: []float64{0, 1, 1, 0},
			want:  0.5,
		},
		{
			name:    "Empty vectors",
			yTrue:   []float64{},
			yPred:   []float64{},
			wantErr: true,
		},
		{
			name:    "Dimension mismatch",
			yTrue:   []float64{0, 1},
			yPred:   []float64{0},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var yTrue, yPred *mat.VecDense
			if len(tt.yTrue) > 0 {
				yTrue = mat.NewVecDense(len(tt.yTrue), tt.yTrue)
			}
			if len(tt.yPred) > 0 {
				yPred = mat.NewVecDense(len(tt.yPred), tt.yPred)
			}

			got, err := ClassificationError(yTrue, yPred)
			if (err != nil) != tt.wantErr {
				t.Errorf("ClassificationError() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && math.Abs(got-tt.want) > 1e-6 {
				t.Errorf("ClassificationError() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAccuracy(t *testing.T) {
	tests := []struct {
		name    string
		yTrue   []float64
		yPred   []float64
		want    float64
		wantErr bool
	}{
		{
			name:  "Perfect accuracy",
			yTrue: []float64{0, 1, 2, 1, 0},
			yPred: []float64{0, 1, 2, 1, 0},
			want:  1.0,
		},
		{
			name:  "80% accuracy",
			yTrue: []float64{0, 1, 2, 1, 0},
			yPred: []float64{0, 1, 1, 1, 0},
			want:  0.8,
		},
		{
			name:  "Zero accuracy",
			yTrue: []float64{0, 0, 0},
			yPred: []float64{1, 1, 1},
			want:  0.0,
		},
		{
			name:    "Empty vectors",
			yTrue:   []float64{},
			yPred:   []float64{},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var yTrue, yPred *mat.VecDense
			if len(tt.yTrue) > 0 {
				yTrue = mat.NewVecDense(len(tt.yTrue), tt.yTrue)
			}
			if len(tt.yPred) > 0 {
				yPred = mat.NewVecDense(len(tt.yPred), tt.yPred)
			}

			got, err := Accuracy(yTrue, yPred)
			if (err != nil) != tt.wantErr {
				t.Errorf("Accuracy() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && math.Abs(got-tt.want) > 1e-6 {
				t.Errorf("Accuracy() = %v, want %v", got, tt.want)
			}
		})
	}
}

// Benchmark tests
func BenchmarkAUC(b *testing.B) {
	// Create test data
	n := 1000
	yTrue := make([]float64, n)
	yPred := make([]float64, n)
	for i := 0; i < n; i++ {
		if i < n/2 {
			yTrue[i] = 0
			yPred[i] = float64(i) / float64(n)
		} else {
			yTrue[i] = 1
			yPred[i] = float64(i) / float64(n)
		}
	}
	yTrueVec := mat.NewVecDense(n, yTrue)
	yPredVec := mat.NewVecDense(n, yPred)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = AUC(yTrueVec, yPredVec)
	}
}

func BenchmarkBinaryLogLoss(b *testing.B) {
	// Create test data
	n := 1000
	yTrue := make([]float64, n)
	yPred := make([]float64, n)
	for i := 0; i < n; i++ {
		if i < n/2 {
			yTrue[i] = 0
			yPred[i] = 0.1 + 0.3*float64(i)/float64(n)
		} else {
			yTrue[i] = 1
			yPred[i] = 0.6 + 0.3*float64(i-n/2)/float64(n/2)
		}
	}
	yTrueVec := mat.NewVecDense(n, yTrue)
	yPredVec := mat.NewVecDense(n, yPred)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = BinaryLogLoss(yTrueVec, yPredVec)
	}
}
