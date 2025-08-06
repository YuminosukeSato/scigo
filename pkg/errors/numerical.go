package errors

import (
	"math"
)

// CheckNumericalStability checks if values contain NaN or Inf
// and returns an error if numerical instability is detected.
func CheckNumericalStability(operation string, values []float64, iteration int) error {
	for _, v := range values {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return NewNumericalInstabilityError(operation, values, iteration)
		}
	}
	return nil
}

// CheckScalar checks a single scalar value for numerical instability.
func CheckScalar(operation string, value float64, iteration int) error {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return NewNumericalInstabilityError(operation, []float64{value}, iteration)
	}
	return nil
}

// CheckMatrix checks all values in a matrix for numerical instability.
func CheckMatrix(operation string, matrix interface{ At(int, int) float64 }, rows, cols, iteration int) error {
	var unstableValues []float64
	
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			v := matrix.At(i, j)
			if math.IsNaN(v) || math.IsInf(v, 0) {
				unstableValues = append(unstableValues, v)
				if len(unstableValues) >= 10 {
					// Limit the number of collected values for error message
					break
				}
			}
		}
		if len(unstableValues) > 0 {
			break
		}
	}
	
	if len(unstableValues) > 0 {
		return NewNumericalInstabilityError(operation, unstableValues, iteration)
	}
	
	return nil
}

// SafeDivide performs division with protection against division by zero.
// Returns 0 if denominator is zero or close to zero.
func SafeDivide(numerator, denominator float64) float64 {
	if math.Abs(denominator) < 1e-10 {
		return 0
	}
	return numerator / denominator
}

// ClipValue clips a value to the range [min, max].
func ClipValue(value, min, max float64) float64 {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}

// ClipGradient clips gradient values to prevent explosion.
func ClipGradient(gradient []float64, maxNorm float64) []float64 {
	// Calculate L2 norm
	var norm float64
	for _, g := range gradient {
		norm += g * g
	}
	norm = math.Sqrt(norm)
	
	// If norm exceeds maxNorm, scale down
	if norm > maxNorm {
		scale := maxNorm / norm
		clipped := make([]float64, len(gradient))
		for i, g := range gradient {
			clipped[i] = g * scale
		}
		return clipped
	}
	
	return gradient
}

// StabilizeLog computes log with protection against log(0).
// Returns log(max(value, epsilon)) where epsilon is a small positive number.
func StabilizeLog(value float64) float64 {
	const epsilon = 1e-10
	if value < epsilon {
		return math.Log(epsilon)
	}
	return math.Log(value)
}

// StabilizeExp computes exp with protection against overflow.
// Clips the input to prevent exp from returning Inf.
func StabilizeExp(value float64) float64 {
	const maxExp = 700.0 // exp(700) is close to the maximum float64
	if value > maxExp {
		return math.Exp(maxExp)
	}
	if value < -maxExp {
		return 0
	}
	return math.Exp(value)
}

// LogSumExp computes log(sum(exp(values))) in a numerically stable way.
func LogSumExp(values []float64) float64 {
	if len(values) == 0 {
		return math.Inf(-1)
	}
	
	// Find maximum value
	maxVal := values[0]
	for _, v := range values[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	
	// If max is -Inf, all values are -Inf
	if math.IsInf(maxVal, -1) {
		return math.Inf(-1)
	}
	
	// Compute sum(exp(v - max))
	sum := 0.0
	for _, v := range values {
		sum += math.Exp(v - maxVal)
	}
	
	return maxVal + math.Log(sum)
}