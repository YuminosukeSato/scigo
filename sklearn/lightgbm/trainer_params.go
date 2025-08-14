package lightgbm

import (
	"math/rand"
)

// SamplingStrategy handles data and feature sampling for training
type SamplingStrategy struct {
	rng              *rand.Rand
	featureFraction  float64
	baggingFraction  float64
	baggingFreq      int
	deterministic    bool
	sampledFeatures  []int
	sampledInstances []int
}

// NewSamplingStrategy creates a new sampling strategy
func NewSamplingStrategy(params TrainingParams) *SamplingStrategy {
	seed := params.Seed
	if seed == 0 && !params.Deterministic {
		seed = int(rand.Int31())
	}

	return &SamplingStrategy{
		rng:             rand.New(rand.NewSource(int64(seed))),
		featureFraction: params.FeatureFraction,
		baggingFraction: params.BaggingFraction,
		baggingFreq:     params.BaggingFreq,
		deterministic:   params.Deterministic,
	}
}

// SampleFeatures samples features for tree building
func (s *SamplingStrategy) SampleFeatures(numFeatures int, iteration int) []int {
	// If feature fraction is 1.0, use all features
	if s.featureFraction >= 1.0 || s.featureFraction <= 0 {
		features := make([]int, numFeatures)
		for i := 0; i < numFeatures; i++ {
			features[i] = i
		}
		return features
	}

	// Calculate number of features to sample
	numSample := int(float64(numFeatures) * s.featureFraction)
	if numSample < 1 {
		numSample = 1
	}
	if numSample > numFeatures {
		numSample = numFeatures
	}

	// Create permutation with deterministic seed if needed
	if s.deterministic {
		// Use iteration as additional seed component for deterministic behavior
		s.rng.Seed(int64(s.rng.Int31() + int32(iteration)))
	}

	// Fisher-Yates shuffle to sample features
	perm := make([]int, numFeatures)
	for i := 0; i < numFeatures; i++ {
		perm[i] = i
	}

	for i := 0; i < numSample; i++ {
		j := i + s.rng.Intn(numFeatures-i)
		perm[i], perm[j] = perm[j], perm[i]
	}

	return perm[:numSample]
}

// SampleInstances samples training instances for tree building
func (s *SamplingStrategy) SampleInstances(numInstances int, iteration int) []int {
	// Check if bagging should be performed this iteration
	if s.baggingFreq <= 0 || iteration%s.baggingFreq != 0 {
		// No bagging this iteration, use all instances
		instances := make([]int, numInstances)
		for i := 0; i < numInstances; i++ {
			instances[i] = i
		}
		return instances
	}

	// If bagging fraction is 1.0, use all instances
	if s.baggingFraction >= 1.0 || s.baggingFraction <= 0 {
		instances := make([]int, numInstances)
		for i := 0; i < numInstances; i++ {
			instances[i] = i
		}
		return instances
	}

	// Calculate number of instances to sample
	numSample := int(float64(numInstances) * s.baggingFraction)
	if numSample < 1 {
		numSample = 1
	}
	if numSample > numInstances {
		numSample = numInstances
	}

	// Create permutation with deterministic seed if needed
	if s.deterministic {
		// Use iteration as additional seed component for deterministic behavior
		s.rng.Seed(int64(s.rng.Int31() + int32(iteration)))
	}

	// Sample without replacement
	perm := make([]int, numInstances)
	for i := 0; i < numInstances; i++ {
		perm[i] = i
	}

	for i := 0; i < numSample; i++ {
		j := i + s.rng.Intn(numInstances-i)
		perm[i], perm[j] = perm[j], perm[i]
	}

	return perm[:numSample]
}

// RegularizationStrategy handles L1/L2 regularization
type RegularizationStrategy struct {
	lambdaL1 float64
	lambdaL2 float64
}

// NewRegularizationStrategy creates a new regularization strategy
func NewRegularizationStrategy(params TrainingParams) *RegularizationStrategy {
	return &RegularizationStrategy{
		lambdaL1: params.Alpha,
		lambdaL2: params.Lambda,
	}
}

// ApplyLeafRegularization applies L1/L2 regularization to leaf value calculation
func (r *RegularizationStrategy) ApplyLeafRegularization(sumGrad, sumHess float64) float64 {
	const epsilon = 1e-10

	// Apply L2 regularization
	denominator := sumHess + r.lambdaL2 + epsilon

	// Apply L1 regularization (soft thresholding)
	if r.lambdaL1 > 0 {
		if sumGrad > r.lambdaL1 {
			return -(sumGrad - r.lambdaL1) / denominator
		} else if sumGrad < -r.lambdaL1 {
			return -(sumGrad + r.lambdaL1) / denominator
		} else {
			return 0.0
		}
	}

	return -sumGrad / denominator
}

// CalculateSplitGain calculates the gain for a split with regularization
func (r *RegularizationStrategy) CalculateSplitGain(
	leftGrad, leftHess, rightGrad, rightHess, parentGrad, parentHess float64) float64 {

	const epsilon = 1e-10

	// Calculate scores with L2 regularization
	leftScore := r.calculateScore(leftGrad, leftHess)
	rightScore := r.calculateScore(rightGrad, rightHess)
	parentScore := r.calculateScore(parentGrad, parentHess)

	// Gain = left_score + right_score - parent_score
	return leftScore + rightScore - parentScore
}

// calculateScore calculates the score for a node with regularization
func (r *RegularizationStrategy) calculateScore(sumGrad, sumHess float64) float64 {
	const epsilon = 1e-10

	// Apply L2 regularization
	denominator := sumHess + r.lambdaL2 + epsilon

	// Apply L1 regularization
	var numerator float64
	if r.lambdaL1 > 0 {
		if sumGrad > r.lambdaL1 {
			numerator = sumGrad - r.lambdaL1
		} else if sumGrad < -r.lambdaL1 {
			numerator = sumGrad + r.lambdaL1
		} else {
			return 0.0
		}
	} else {
		numerator = sumGrad
	}

	// Score = -0.5 * G^2 / (H + lambda)
	return 0.5 * numerator * numerator / denominator
}
