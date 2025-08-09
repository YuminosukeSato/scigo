package lightgbm

import (
	"math"
	"sort"
)

// CategoricalSplit represents a categorical feature split
type CategoricalSplit struct {
	Feature    int     // Feature index
	Categories []int   // Categories that go to the left child
	Gain       float64 // Split gain
	LeftGrad   float64 // Left gradient sum
	RightGrad  float64 // Right gradient sum
	LeftHess   float64 // Left hessian sum
	RightHess  float64 // Right hessian sum
	LeftCount  int     // Left sample count
	RightCount int     // Right sample count
}

// CategoryInfo stores information about a category
type CategoryInfo struct {
	Category int
	Count    int
	SumGrad  float64
	SumHess  float64
}

// isCategoricalFeature checks if a feature is categorical
func (t *Trainer) isCategoricalFeature(featureIdx int) bool {
	for _, catIdx := range t.params.CategoricalFeatures {
		if catIdx == featureIdx {
			return true
		}
	}
	return false
}

// findBestCategoricalSplit finds the best split for a categorical feature
func (t *Trainer) findBestCategoricalSplit(indices []int, feature int) SplitInfo {
	// Collect category statistics
	categoryStats := make(map[int]*CategoryInfo)

	for _, idx := range indices {
		category := int(t.X.At(idx, feature))

		if stats, exists := categoryStats[category]; exists {
			stats.Count++
			stats.SumGrad += t.gradients[idx]
			stats.SumHess += t.hessians[idx]
		} else {
			categoryStats[category] = &CategoryInfo{
				Category: category,
				Count:    1,
				SumGrad:  t.gradients[idx],
				SumHess:  t.hessians[idx],
			}
		}
	}

	// Convert to slice for processing
	categories := make([]*CategoryInfo, 0, len(categoryStats))
	for _, info := range categoryStats {
		categories = append(categories, info)
	}

	// If only one category, no split possible
	if len(categories) <= 1 {
		return SplitInfo{
			Feature: feature,
			Gain:    -math.MaxFloat64,
		}
	}

	// Sort categories by gradient/hessian ratio (for greedy approach)
	sort.Slice(categories, func(i, j int) bool {
		ratioI := categories[i].SumGrad / (categories[i].SumHess + t.params.Lambda)
		ratioJ := categories[j].SumGrad / (categories[j].SumHess + t.params.Lambda)
		return ratioI < ratioJ
	})

	// Find best split
	bestSplit := SplitInfo{
		Feature: feature,
		Gain:    -math.MaxFloat64,
	}

	// Calculate total gradient and hessian
	totalGrad := 0.0
	totalHess := 0.0
	for _, cat := range categories {
		totalGrad += cat.SumGrad
		totalHess += cat.SumHess
	}

	// Try different split points
	// For simplicity, use greedy approach: try splitting after each category
	leftGrad := 0.0
	leftHess := 0.0
	leftCount := 0
	leftCategories := []int{}

	for i := 0; i < len(categories)-1; i++ {
		leftGrad += categories[i].SumGrad
		leftHess += categories[i].SumHess
		leftCount += categories[i].Count
		leftCategories = append(leftCategories, categories[i].Category)

		rightGrad := totalGrad - leftGrad
		rightHess := totalHess - leftHess
		rightCount := len(indices) - leftCount

		// Check minimum data constraints
		if leftCount < t.params.MinDataInLeaf || rightCount < t.params.MinDataInLeaf {
			continue
		}

		// Calculate gain
		gain := t.calculateSplitGain(leftGrad, leftHess, rightGrad, rightHess, totalGrad, totalHess)

		if gain > bestSplit.Gain {
			bestSplit.Gain = gain
			bestSplit.LeftCount = leftCount
			bestSplit.RightCount = rightCount
			bestSplit.LeftGrad = leftGrad
			bestSplit.RightGrad = rightGrad
			bestSplit.LeftHess = leftHess
			bestSplit.RightHess = rightHess
			// Store categories for left split (will be stored in Node.Categories)
			bestSplit.Threshold = float64(len(leftCategories)) // Store count as placeholder
		}
	}

	// If we have too many categories, consider one-hot encoding for small sets
	if len(categories) <= t.params.MaxCatToOnehot && t.params.MaxCatToOnehot > 0 {
		// Try one-vs-rest splits
		for _, targetCat := range categories {
			leftGrad = targetCat.SumGrad
			leftHess = targetCat.SumHess
			leftCount = targetCat.Count

			rightGrad := totalGrad - leftGrad
			rightHess := totalHess - leftHess
			rightCount := len(indices) - leftCount

			if leftCount < t.params.MinDataInLeaf || rightCount < t.params.MinDataInLeaf {
				continue
			}

			gain := t.calculateSplitGain(leftGrad, leftHess, rightGrad, rightHess, totalGrad, totalHess)

			if gain > bestSplit.Gain {
				bestSplit.Gain = gain
				bestSplit.LeftCount = leftCount
				bestSplit.RightCount = rightCount
				bestSplit.LeftGrad = leftGrad
				bestSplit.RightGrad = rightGrad
				bestSplit.LeftHess = leftHess
				bestSplit.RightHess = rightHess
				bestSplit.Threshold = float64(targetCat.Category) // Store single category
			}
		}
	}

	return bestSplit
}

// splitCategoricalData splits indices based on a categorical split
func (t *Trainer) splitCategoricalData(indices []int, feature int, leftCategories map[int]bool) ([]int, []int) {
	var leftIndices, rightIndices []int

	for _, idx := range indices {
		category := int(t.X.At(idx, feature))
		if leftCategories[category] {
			leftIndices = append(leftIndices, idx)
		} else {
			rightIndices = append(rightIndices, idx)
		}
	}

	return leftIndices, rightIndices
}

// getCategoriesForSplit extracts the categories for a categorical split
func (t *Trainer) getCategoriesForSplit(indices []int, feature int, splitInfo SplitInfo) []int {
	// Collect category statistics
	categoryStats := make(map[int]*CategoryInfo)

	for _, idx := range indices {
		category := int(t.X.At(idx, feature))

		if stats, exists := categoryStats[category]; exists {
			stats.Count++
			stats.SumGrad += t.gradients[idx]
			stats.SumHess += t.hessians[idx]
		} else {
			categoryStats[category] = &CategoryInfo{
				Category: category,
				Count:    1,
				SumGrad:  t.gradients[idx],
				SumHess:  t.hessians[idx],
			}
		}
	}

	// Convert to slice and sort
	categories := make([]*CategoryInfo, 0, len(categoryStats))
	for _, info := range categoryStats {
		categories = append(categories, info)
	}

	sort.Slice(categories, func(i, j int) bool {
		ratioI := categories[i].SumGrad / (categories[i].SumHess + t.params.Lambda)
		ratioJ := categories[j].SumGrad / (categories[j].SumHess + t.params.Lambda)
		return ratioI < ratioJ
	})

	// Extract the left categories based on the split threshold (which stores the count)
	numLeftCategories := int(splitInfo.Threshold)
	if numLeftCategories == 1 {
		// Single category split (one-hot)
		return []int{int(splitInfo.Threshold)}
	}

	leftCategories := make([]int, numLeftCategories)
	for i := 0; i < numLeftCategories && i < len(categories); i++ {
		leftCategories[i] = categories[i].Category
	}

	return leftCategories
}
