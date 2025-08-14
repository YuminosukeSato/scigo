package lightgbm

import (
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
	if numLeftCategories == 1 && len(categories) > 0 {
		// Single category split - return the first category after sorting
		return []int{categories[0].Category}
	}

	leftCategories := make([]int, numLeftCategories)
	for i := 0; i < numLeftCategories && i < len(categories); i++ {
		leftCategories[i] = categories[i].Category
	}

	return leftCategories
}
