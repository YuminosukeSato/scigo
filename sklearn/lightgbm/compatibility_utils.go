package lightgbm

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// ULPDistance calculates the Unit in the Last Place distance between two float32 values.
// This provides exact comparison of floating point numbers.
func ULPDistance32(a, b float32) uint32 {
	if a == b {
		return 0
	}

	// Handle special cases
	if math.IsNaN(float64(a)) || math.IsNaN(float64(b)) {
		return math.MaxUint32
	}
	if math.IsInf(float64(a), 0) || math.IsInf(float64(b), 0) {
		if a == b {
			return 0
		}
		return math.MaxUint32
	}

	// Convert to integer representation
	aBits := math.Float32bits(a)
	bBits := math.Float32bits(b)

	// Make lexicographically ordered as twos-complement
	if aBits&0x80000000 != 0 {
		aBits = 0x80000000 - aBits
	}
	if bBits&0x80000000 != 0 {
		bBits = 0x80000000 - bBits
	}

	// Calculate distance
	if aBits > bBits {
		return aBits - bBits
	}
	return bBits - aBits
}

// ULPDistance64 calculates the ULP distance for float64 values
func ULPDistance64(a, b float64) uint64 {
	if a == b {
		return 0
	}

	// Handle special cases
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.MaxUint64
	}
	if math.IsInf(a, 0) || math.IsInf(b, 0) {
		if a == b {
			return 0
		}
		return math.MaxUint64
	}

	// Convert to integer representation
	aBits := math.Float64bits(a)
	bBits := math.Float64bits(b)

	// Make lexicographically ordered as twos-complement
	if aBits&0x8000000000000000 != 0 {
		aBits = 0x8000000000000000 - aBits
	}
	if bBits&0x8000000000000000 != 0 {
		bBits = 0x8000000000000000 - bBits
	}

	// Calculate distance
	if aBits > bBits {
		return aBits - bBits
	}
	return bBits - aBits
}

// AlmostEqual32 checks if two float32 values are almost equal within maxULP units
func AlmostEqual32(a, b float32, maxULP uint32) bool {
	return ULPDistance32(a, b) <= maxULP
}

// AlmostEqual64 checks if two float64 values are almost equal within maxULP units
func AlmostEqual64(a, b float64, maxULP uint64) bool {
	return ULPDistance64(a, b) <= maxULP
}

// LoadBinaryArray loads array data from binary file format matching Python's output
func LoadBinaryArray(filepath string) ([]float32, []int, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	// Read shape information
	var ndim uint32
	if err := binary.Read(file, binary.LittleEndian, &ndim); err != nil {
		return nil, nil, fmt.Errorf("failed to read ndim: %w", err)
	}

	shape := make([]int, ndim)
	for i := range shape {
		var dim uint32
		if err := binary.Read(file, binary.LittleEndian, &dim); err != nil {
			return nil, nil, fmt.Errorf("failed to read shape[%d]: %w", i, err)
		}
		shape[i] = int(dim)
	}

	// Read data type
	var dtype uint32
	if err := binary.Read(file, binary.LittleEndian, &dtype); err != nil {
		return nil, nil, fmt.Errorf("failed to read dtype: %w", err)
	}

	// Calculate total elements
	totalElements := 1
	for _, dim := range shape {
		totalElements *= dim
	}

	// Read data based on type
	if dtype == 0 { // float32
		data := make([]float32, totalElements)
		if err := binary.Read(file, binary.LittleEndian, data); err != nil {
			return nil, nil, fmt.Errorf("failed to read float32 data: %w", err)
		}
		return data, shape, nil
	}

	return nil, nil, fmt.Errorf("unsupported dtype: %d", dtype)
}

// CompareArrays compares two float32 arrays and returns detailed comparison results
func CompareArrays(a, b []float32, maxULP uint32) (*ArrayComparisonResult, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("array lengths differ: %d vs %d", len(a), len(b))
	}

	result := &ArrayComparisonResult{
		Length:       len(a),
		MaxULP:       maxULP,
		ExactMatches: 0,
		Differences:  make([]DifferenceDetail, 0),
	}

	var maxDiff float32
	var maxULPFound uint32

	for i := range a {
		ulp := ULPDistance32(a[i], b[i])

		if ulp == 0 {
			result.ExactMatches++
		} else if ulp > maxULP {
			// Record significant differences
			result.Differences = append(result.Differences, DifferenceDetail{
				Index:    i,
				Expected: a[i],
				Actual:   b[i],
				ULP:      ulp,
			})
		}

		diff := float32(math.Abs(float64(a[i] - b[i])))
		if diff > maxDiff {
			maxDiff = diff
			result.MaxAbsDiff = maxDiff
		}

		if ulp > maxULPFound {
			maxULPFound = ulp
			result.MaxULPFound = maxULPFound
		}
	}

	result.AllMatch = len(result.Differences) == 0
	return result, nil
}

// ArrayComparisonResult contains detailed comparison results
type ArrayComparisonResult struct {
	Length       int
	MaxULP       uint32
	MaxULPFound  uint32
	MaxAbsDiff   float32
	ExactMatches int
	AllMatch     bool
	Differences  []DifferenceDetail
}

// DifferenceDetail describes a single difference between arrays
type DifferenceDetail struct {
	Index    int
	Expected float32
	Actual   float32
	ULP      uint32
}

// String formats the comparison result for display
func (r *ArrayComparisonResult) String() string {
	if r.AllMatch {
		return fmt.Sprintf("✓ Arrays match perfectly (%d elements, %d exact matches)",
			r.Length, r.ExactMatches)
	}

	return fmt.Sprintf("✗ Arrays differ:\n"+
		"  Length: %d\n"+
		"  Exact matches: %d (%.1f%%)\n"+
		"  Max ULP distance: %d (threshold: %d)\n"+
		"  Max absolute diff: %e\n"+
		"  Differences found: %d",
		r.Length,
		r.ExactMatches, float64(r.ExactMatches)*100.0/float64(r.Length),
		r.MaxULPFound, r.MaxULP,
		r.MaxAbsDiff,
		len(r.Differences))
}

// PrintDifferences prints detailed difference information
func (r *ArrayComparisonResult) PrintDifferences(w io.Writer, maxShow int) {
	if len(r.Differences) == 0 {
		return
	}

	fmt.Fprintf(w, "\nFirst %d differences:\n", min(maxShow, len(r.Differences)))
	for i, diff := range r.Differences {
		if i >= maxShow {
			break
		}
		fmt.Fprintf(w, "  [%d]: expected=%g, actual=%g, ULP=%d\n",
			diff.Index, diff.Expected, diff.Actual, diff.ULP)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

