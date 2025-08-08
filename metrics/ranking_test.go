package metrics

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestNDCG(t *testing.T) {
	tests := []struct {
		name    string
		yTrue   []float64
		yPred   []float64
		k       int
		want    float64
		wantErr bool
	}{
		{
			name:  "Perfect ranking",
			yTrue: []float64{3, 2, 3, 0, 1, 2},
			yPred: []float64{3.1, 2.9, 3.0, 0.1, 1.1, 2.1}, // Perfect order with scores
			k:     -1,
			want:  1.0,
		},
		{
			name:  "Worst ranking",
			yTrue: []float64{3, 2, 3, 0, 1, 2},
			yPred: []float64{1, 2, 3, 4, 5, 6}, // Reverse order
			k:     -1,
			want:  0.706, // Corrected expected value
		},
		{
			name:  "NDCG@3",
			yTrue: []float64{3, 2, 3, 0, 1, 2},
			yPred: []float64{2.5, 0.5, 2, 0, 1, 3},
			k:     3,
			want:  0.845, // Corrected expected value
		},
		{
			name:  "Binary relevance",
			yTrue: []float64{1, 0, 1, 0, 1},
			yPred: []float64{0.9, 0.8, 0.7, 0.6, 0.5},
			k:     -1,
			want:  0.885, // Corrected expected value
		},
		{
			name:  "All zeros relevance",
			yTrue: []float64{0, 0, 0, 0},
			yPred: []float64{1, 2, 3, 4},
			k:     -1,
			want:  0.0,
		},
		{
			name:  "Single element",
			yTrue: []float64{2},
			yPred: []float64{1},
			k:     1,
			want:  1.0, // Only one element, so it's perfect
		},
		{
			name:    "Negative relevance",
			yTrue:   []float64{1, -1, 2},
			yPred:   []float64{1, 2, 3},
			k:       -1,
			wantErr: true,
		},
		{
			name:    "Dimension mismatch",
			yTrue:   []float64{1, 2, 3},
			yPred:   []float64{1, 2},
			k:       -1,
			wantErr: true,
		},
		{
			name:    "Invalid k",
			yTrue:   []float64{1, 2, 3},
			yPred:   []float64{1, 2, 3},
			k:       0,
			wantErr: true,
		},
		{
			name:    "Empty vectors",
			yTrue:   []float64{},
			yPred:   []float64{},
			k:       1,
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

			got, err := NDCG(yTrue, yPred, tt.k)
			if (err != nil) != tt.wantErr {
				t.Errorf("NDCG() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && math.Abs(got-tt.want) > 0.01 {
				t.Errorf("NDCG() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNDCGMatrix(t *testing.T) {
	tests := []struct {
		name    string
		yTrue   mat.Matrix
		yPred   mat.Matrix
		k       int
		want    float64
		wantErr bool
	}{
		{
			name:  "Matrix input",
			yTrue: mat.NewDense(4, 1, []float64{3, 2, 1, 0}),
			yPred: mat.NewDense(4, 1, []float64{2.5, 2.0, 1.5, 1.0}),
			k:     -1,
			want:  1.0,
		},
		{
			name:  "Multi-column matrix (uses first column)",
			yTrue: mat.NewDense(4, 2, []float64{3, 9, 2, 9, 1, 9, 0, 9}),
			yPred: mat.NewDense(4, 2, []float64{2.5, 9, 2.0, 9, 1.5, 9, 1.0, 9}),
			k:     -1,
			want:  1.0,
		},
		{
			name:    "Nil matrix",
			yTrue:   nil,
			yPred:   mat.NewDense(1, 1, []float64{0.5}),
			k:       1,
			wantErr: true,
		},
		{
			name:    "Empty matrix",
			yTrue:   &mat.Dense{},
			yPred:   &mat.Dense{},
			k:       1,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NDCGMatrix(tt.yTrue, tt.yPred, tt.k)
			if (err != nil) != tt.wantErr {
				t.Errorf("NDCGMatrix() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && math.Abs(got-tt.want) > 0.01 {
				t.Errorf("NDCGMatrix() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAveragePrecision(t *testing.T) {
	tests := []struct {
		name    string
		yTrue   []float64
		yPred   []float64
		want    float64
		wantErr bool
	}{
		{
			name:  "Perfect ranking",
			yTrue: []float64{1, 1, 1, 0, 0},
			yPred: []float64{5, 4, 3, 2, 1},
			want:  1.0,
		},
		{
			name:  "Worst ranking",
			yTrue: []float64{1, 1, 1, 0, 0},
			yPred: []float64{1, 2, 3, 4, 5},
			want:  0.478, // Corrected: (1/3 + 2/4 + 3/5) / 3
		},
		{
			name:  "Mixed ranking",
			yTrue: []float64{1, 0, 1, 0, 1},
			yPred: []float64{0.9, 0.8, 0.7, 0.6, 0.5},
			want:  0.756, // Corrected: (1/1 + 2/3 + 3/5) / 3
		},
		{
			name:  "Single relevant",
			yTrue: []float64{0, 0, 1, 0, 0},
			yPred: []float64{0.1, 0.2, 0.3, 0.4, 0.5},
			want:  0.333, // 1/3
		},
		{
			name:  "No relevant items",
			yTrue: []float64{0, 0, 0, 0},
			yPred: []float64{1, 2, 3, 4},
			want:  0.0,
		},
		{
			name:  "All relevant",
			yTrue: []float64{1, 1, 1},
			yPred: []float64{3, 2, 1},
			want:  1.0,
		},
		{
			name:    "Non-binary labels",
			yTrue:   []float64{0, 0.5, 1},
			yPred:   []float64{1, 2, 3},
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

			got, err := AveragePrecision(yTrue, yPred)
			if (err != nil) != tt.wantErr {
				t.Errorf("AveragePrecision() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && math.Abs(got-tt.want) > 0.01 {
				t.Errorf("AveragePrecision() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMeanAveragePrecision(t *testing.T) {
	tests := []struct {
		name      string
		yTrueList [][]float64
		yPredList [][]float64
		want      float64
		wantErr   bool
	}{
		{
			name: "Multiple queries",
			yTrueList: [][]float64{
				{1, 1, 0, 0},
				{0, 1, 1, 0},
				{1, 0, 0, 1},
			},
			yPredList: [][]float64{
				{4, 3, 2, 1},
				{1, 2, 3, 4},
				{3, 2, 1, 4},
			},
			want: 0.861, // Corrected average of individual APs
		},
		{
			name: "Single query",
			yTrueList: [][]float64{
				{1, 0, 1, 0},
			},
			yPredList: [][]float64{
				{4, 3, 2, 1},
			},
			want: 0.833,
		},
		{
			name:      "Empty lists",
			yTrueList: [][]float64{},
			yPredList: [][]float64{},
			wantErr:   true,
		},
		{
			name: "Mismatched list sizes",
			yTrueList: [][]float64{
				{1, 0},
				{0, 1},
			},
			yPredList: [][]float64{
				{1, 2},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var yTrueList, yPredList []*mat.VecDense

			for _, y := range tt.yTrueList {
				if len(y) > 0 {
					yTrueList = append(yTrueList, mat.NewVecDense(len(y), y))
				}
			}

			for _, y := range tt.yPredList {
				if len(y) > 0 {
					yPredList = append(yPredList, mat.NewVecDense(len(y), y))
				}
			}

			got, err := MeanAveragePrecision(yTrueList, yPredList)
			if (err != nil) != tt.wantErr {
				t.Errorf("MeanAveragePrecision() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && math.Abs(got-tt.want) > 0.01 {
				t.Errorf("MeanAveragePrecision() = %v, want %v", got, tt.want)
			}
		})
	}
}

// Benchmark tests
func BenchmarkNDCG(b *testing.B) {
	// Create test data
	n := 1000
	yTrue := make([]float64, n)
	yPred := make([]float64, n)
	for i := 0; i < n; i++ {
		yTrue[i] = float64(n-i) / float64(n) * 3 // Relevance scores 0-3
		yPred[i] = float64(i) / float64(n)
	}
	yTrueVec := mat.NewVecDense(n, yTrue)
	yPredVec := mat.NewVecDense(n, yPred)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = NDCG(yTrueVec, yPredVec, 10)
	}
}

func BenchmarkAveragePrecision(b *testing.B) {
	// Create test data
	n := 1000
	yTrue := make([]float64, n)
	yPred := make([]float64, n)
	for i := 0; i < n; i++ {
		if i%3 == 0 {
			yTrue[i] = 1
		}
		yPred[i] = float64(i) / float64(n)
	}
	yTrueVec := mat.NewVecDense(n, yTrue)
	yPredVec := mat.NewVecDense(n, yPred)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = AveragePrecision(yTrueVec, yPredVec)
	}
}

func TestDCGCalculation(t *testing.T) {
	// Test the internal DCG calculation
	tests := []struct {
		name      string
		relevance []float64
		want      float64
	}{
		{
			name:      "Basic DCG",
			relevance: []float64{3, 2, 3, 0, 1, 2},
			want:      13.848, // Expected DCG value
		},
		{
			name:      "Binary relevance",
			relevance: []float64{1, 1, 0, 0, 1},
			want:      2.018, // Corrected DCG value
		},
		{
			name:      "All zeros",
			relevance: []float64{0, 0, 0, 0},
			want:      0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create pairs with relevance as both score and relevance (ideal ordering)
			pairs := make([]struct {
				score     float64
				relevance float64
			}, len(tt.relevance))
			for i, rel := range tt.relevance {
				pairs[i] = struct {
					score     float64
					relevance float64
				}{score: rel, relevance: rel}
			}

			got := dcg(pairs, len(pairs))
			if math.Abs(got-tt.want) > 0.01 {
				t.Errorf("dcg() = %v, want %v", got, tt.want)
			}
		})
	}
}
