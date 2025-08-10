package lightgbm

import (
	"os"
	"path/filepath"
	"testing"
)

func TestULPDistance(t *testing.T) {
	tests := []struct {
		name     string
		a, b     float32
		expected uint32
	}{
		{"exact same", 1.0, 1.0, 0},
		{"next representable", 1.0, 1.0000001, 1},
		{"small difference", 1.0, 1.0000002, 2},
		{"zero and small", 0.0, 1e-45, 1},
		{"negative values", -1.0, -1.0000001, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ulp := ULPDistance32(tt.a, tt.b)
			if ulp != tt.expected {
				t.Errorf("ULPDistance32(%g, %g) = %d, want %d", tt.a, tt.b, ulp, tt.expected)
			}
		})
	}
}

func TestLoadGoldenData(t *testing.T) {
	// Check if golden data exists
	goldenDir := "../../tests/compatibility/golden_data"
	if _, err := os.Stat(goldenDir); os.IsNotExist(err) {
		t.Skip("Golden data not found. Run: python3 tests/compatibility/generate_golden_data.py")
	}

	// Try to load X data
	xPath := filepath.Join(goldenDir, "minimal_regression_X.bin")
	data, shape, err := LoadBinaryArray(xPath)
	if err != nil {
		t.Fatalf("Failed to load golden data: %v", err)
	}

	// Verify shape
	if len(shape) != 2 {
		t.Errorf("Expected 2D array, got %d dimensions", len(shape))
	}
	if shape[0] != 20 || shape[1] != 3 {
		t.Errorf("Expected shape [20, 3], got %v", shape)
	}

	// Verify data size
	expectedSize := shape[0] * shape[1]
	if len(data) != expectedSize {
		t.Errorf("Expected %d elements, got %d", expectedSize, len(data))
	}

	// Basic sanity check on values
	for i, v := range data {
		if v < -10 || v > 10 {
			t.Errorf("Unexpected value at index %d: %g", i, v)
		}
	}
}

func TestArrayComparison(t *testing.T) {
	a := []float32{1.0, 2.0, 3.0, 4.0}
	b := []float32{1.0, 2.0000001, 3.0, 4.0000002}

	result, err := CompareArrays(a, b, 2)
	if err != nil {
		t.Fatalf("CompareArrays failed: %v", err)
	}

	if !result.AllMatch {
		t.Errorf("Expected arrays to match within 2 ULP")
	}

	if result.ExactMatches != 2 {
		t.Errorf("Expected 2 exact matches, got %d", result.ExactMatches)
	}

	// Test with stricter tolerance
	result, err = CompareArrays(a, b, 1)
	if err != nil {
		t.Fatalf("CompareArrays failed: %v", err)
	}

	if result.AllMatch {
		t.Errorf("Expected arrays to NOT match within 1 ULP")
	}

	if len(result.Differences) != 1 {
		t.Errorf("Expected 1 difference, got %d", len(result.Differences))
	}
}

