package lightgbm

import (
	"math"
	"sort"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

// CAPITree represents a decision tree in the ensemble for C API compatibility
type CAPITree struct {
	Root *CAPITreeNode
}

// CAPITreeNode represents a node in the decision tree for C API compatibility
type CAPITreeNode struct {
	IsLeaf       bool
	SplitFeature int
	Threshold    float64
	LeafValue    float64
	LeftChild    *CAPITreeNode
	RightChild   *CAPITreeNode
	// For debugging and compatibility testing
	Gain      float64
	DataCount int
	SumGrad   float64
	SumHess   float64
}

// predict returns the leaf value for a given sample
func (t *CAPITree) predict(sample []float64) float64 {
	if t.Root == nil {
		return 0.0
	}
	return t.Root.predict(sample)
}

func (n *CAPITreeNode) predict(sample []float64) float64 {
	if n.IsLeaf {
		return n.LeafValue
	}

	if sample[n.SplitFeature] <= n.Threshold {
		if n.LeftChild != nil {
			return n.LeftChild.predict(sample)
		}
	} else {
		if n.RightChild != nil {
			return n.RightChild.predict(sample)
		}
	}

	return n.LeafValue
}

// buildTree constructs a decision tree using gradients and hessians
func buildTree(data *mat.Dense, gradients, hessians []float64, params map[string]string) CAPITree {
	maxDepth, _ := strconv.Atoi(params["max_depth"])
	minDataInLeaf, _ := strconv.Atoi(params["min_data_in_leaf"])
	numLeaves, _ := strconv.Atoi(params["num_leaves"])
	lambdaL2, _ := strconv.ParseFloat(params["lambda_l2"], 64)
	minGainToSplit, _ := strconv.ParseFloat(params["min_gain_to_split"], 64)

	if maxDepth <= 0 {
		maxDepth = 10 // Default max depth
	}
	if minDataInLeaf <= 0 {
		minDataInLeaf = 20
	}
	if numLeaves <= 0 {
		numLeaves = 31
	}

	nrow, ncol := data.Dims()
	indices := make([]int, nrow)
	for i := range indices {
		indices[i] = i
	}

	// Build tree recursively
	root := buildNode(
		data,
		indices,
		gradients,
		hessians,
		0, // depth
		maxDepth,
		minDataInLeaf,
		lambdaL2,
		minGainToSplit,
		ncol,
	)

	return CAPITree{Root: root}
}

func buildNode(
	data *mat.Dense,
	indices []int,
	gradients, hessians []float64,
	depth, maxDepth, minDataInLeaf int,
	lambdaL2, minGainToSplit float64,
	numFeatures int,
) *CAPITreeNode {
	// Calculate sum of gradients and hessians for this node
	sumGrad := 0.0
	sumHess := 0.0
	for _, idx := range indices {
		sumGrad += gradients[idx]
		sumHess += hessians[idx]
	}

	// Check stopping conditions
	if depth >= maxDepth || len(indices) < 2*minDataInLeaf {
		// Create leaf node
		leafValue := -sumGrad / (sumHess + lambdaL2)
		return &CAPITreeNode{
			IsLeaf:    true,
			LeafValue: leafValue,
			DataCount: len(indices),
			SumGrad:   sumGrad,
			SumHess:   sumHess,
		}
	}

	// Find best split
	bestSplit := findBestSplit(
		data,
		indices,
		gradients,
		hessians,
		sumGrad,
		sumHess,
		minDataInLeaf,
		lambdaL2,
		minGainToSplit,
		numFeatures,
	)

	// If no good split found, create leaf
	if bestSplit.gain <= minGainToSplit {
		leafValue := -sumGrad / (sumHess + lambdaL2)
		return &CAPITreeNode{
			IsLeaf:    true,
			LeafValue: leafValue,
			DataCount: len(indices),
			SumGrad:   sumGrad,
			SumHess:   sumHess,
		}
	}

	// Split the data
	leftIndices, rightIndices := splitData(data, indices, bestSplit.feature, bestSplit.threshold)

	// Recursively build children
	leftChild := buildNode(
		data,
		leftIndices,
		gradients,
		hessians,
		depth+1,
		maxDepth,
		minDataInLeaf,
		lambdaL2,
		minGainToSplit,
		numFeatures,
	)

	rightChild := buildNode(
		data,
		rightIndices,
		gradients,
		hessians,
		depth+1,
		maxDepth,
		minDataInLeaf,
		lambdaL2,
		minGainToSplit,
		numFeatures,
	)

	// Create internal node
	return &CAPITreeNode{
		IsLeaf:       false,
		SplitFeature: bestSplit.feature,
		Threshold:    bestSplit.threshold,
		LeftChild:    leftChild,
		RightChild:   rightChild,
		Gain:         bestSplit.gain,
		DataCount:    len(indices),
		SumGrad:      sumGrad,
		SumHess:      sumHess,
	}
}

type splitInfo struct {
	feature   int
	threshold float64
	gain      float64
}

func findBestSplit(
	data *mat.Dense,
	indices []int,
	gradients, hessians []float64,
	totalGrad, totalHess float64,
	minDataInLeaf int,
	lambdaL2, minGainToSplit float64,
	numFeatures int,
) splitInfo {
	bestSplit := splitInfo{gain: -math.MaxFloat64}

	// Try each feature
	for feature := 0; feature < numFeatures; feature++ {
		// Get feature values for the current indices
		values := make([]float64, len(indices))
		for i, idx := range indices {
			values[i] = data.At(idx, feature)
		}

		// Find unique thresholds
		uniqueValues := getUniqueValues(values)
		if len(uniqueValues) <= 1 {
			continue // No split possible
		}

		// Sort indices by feature value
		sortedIndices := make([]int, len(indices))
		copy(sortedIndices, indices)
		sort.Slice(sortedIndices, func(i, j int) bool {
			return data.At(sortedIndices[i], feature) < data.At(sortedIndices[j], feature)
		})

		// Try each possible split threshold
		leftGrad := 0.0
		leftHess := 0.0

		for i := 0; i < len(sortedIndices)-1; i++ {
			idx := sortedIndices[i]
			leftGrad += gradients[idx]
			leftHess += hessians[idx]

			// Check minimum data in leaf
			leftCount := i + 1
			rightCount := len(sortedIndices) - leftCount
			if leftCount < minDataInLeaf || rightCount < minDataInLeaf {
				continue
			}

			// Skip if same value
			currentVal := data.At(sortedIndices[i], feature)
			nextVal := data.At(sortedIndices[i+1], feature)
			if currentVal == nextVal {
				continue
			}

			// Calculate gain
			rightGrad := totalGrad - leftGrad
			rightHess := totalHess - leftHess

			gain := calculateSplitGain(
				leftGrad, leftHess,
				rightGrad, rightHess,
				totalGrad, totalHess,
				lambdaL2,
			)

			if gain > bestSplit.gain {
				bestSplit.feature = feature
				bestSplit.threshold = (currentVal + nextVal) / 2.0
				bestSplit.gain = gain
			}
		}
	}

	return bestSplit
}

func calculateSplitGain(
	leftGrad, leftHess,
	rightGrad, rightHess,
	totalGrad, totalHess,
	lambdaL2 float64,
) float64 {
	// Calculate gain using LightGBM formula
	leftScore := (leftGrad * leftGrad) / (leftHess + lambdaL2)
	rightScore := (rightGrad * rightGrad) / (rightHess + lambdaL2)
	totalScore := (totalGrad * totalGrad) / (totalHess + lambdaL2)

	return 0.5 * (leftScore + rightScore - totalScore)
}

func splitData(data *mat.Dense, indices []int, feature int, threshold float64) ([]int, []int) {
	leftIndices := make([]int, 0)
	rightIndices := make([]int, 0)

	for _, idx := range indices {
		if data.At(idx, feature) <= threshold {
			leftIndices = append(leftIndices, idx)
		} else {
			rightIndices = append(rightIndices, idx)
		}
	}

	return leftIndices, rightIndices
}

func getUniqueValues(values []float64) []float64 {
	unique := make(map[float64]bool)
	for _, v := range values {
		unique[v] = true
	}

	result := make([]float64, 0, len(unique))
	for v := range unique {
		result = append(result, v)
	}

	sort.Float64s(result)
	return result
}

