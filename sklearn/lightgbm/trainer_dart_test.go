package lightgbm

import (
	"testing"
)

// TestDARTSelectDropIndicesDeterministic ensures drop selection is deterministic and bounded
func TestDARTSelectDropIndicesDeterministic(t *testing.T) {
	params := TrainingParams{
		BoostingType:  "dart",
		DropRate:      0.2,
		MaxDrop:       3,
		SkipDrop:      0.0,
		UniformDrop:   true,
		DropSeed:      123,
		NumIterations: 1,
	}
	tr := NewTrainer(params)

	numTrees := 10
	s1 := tr.selectDARTDropIndices(numTrees, 0)
	s2 := tr.selectDARTDropIndices(numTrees, 0)

	if len(s1) == 0 {
		t.Fatalf("expected some trees to be dropped")
	}
	if len(s1) > params.MaxDrop {
		t.Fatalf("drop count exceeds MaxDrop: %d > %d", len(s1), params.MaxDrop)
	}
	// Deterministic with same seed and iteration
	if len(s1) != len(s2) {
		t.Fatalf("non-deterministic drop size: %d vs %d", len(s1), len(s2))
	}
	for i := range s1 {
		if s1[i] != s2[i] {
			t.Fatalf("non-deterministic indices at %d: %d vs %d", i, s1[i], s2[i])
		}
	}
}

// TestDARTNormalizeWeights validates normalization scales kept trees
func TestDARTNormalizeWeights(t *testing.T) {
	params := TrainingParams{BoostingType: "dart", LearningRate: 0.1}
	tr := NewTrainer(params)

	// Create 5 trees with equal shrinkage
	tr.trees = make([]Tree, 5)
	for i := 0; i < 5; i++ {
		tr.trees[i] = Tree{TreeIndex: i, ShrinkageRate: 0.1}
	}

	// Drop indices {1,3}
	dropped := []int{1, 3}
	tr.normalizeDARTWeights(dropped)

	// Factor should be N / (N - K) = 5 / 3
	factor := float64(5) / float64(5-len(dropped))
	for i := range tr.trees {
		if i == 1 || i == 3 {
			// dropped trees untouched
			if tr.trees[i].ShrinkageRate != 0.1 {
				t.Fatalf("dropped tree %d weight changed: %f", i, tr.trees[i].ShrinkageRate)
			}
		} else {
			exp := 0.1 * factor
			if (tr.trees[i].ShrinkageRate-exp) > 1e-12 || (exp-tr.trees[i].ShrinkageRate) > 1e-12 {
				t.Fatalf("kept tree %d not normalized: got %f want %f", i, tr.trees[i].ShrinkageRate, exp)
			}
		}
	}
}
