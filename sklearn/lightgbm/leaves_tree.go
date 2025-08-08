package lightgbm

import (
	"math"
)

// LeavesNode represents a node in the leaves-compatible tree structure
type LeavesNode struct {
	Threshold float64
	Left      uint32
	Right     uint32
	Feature   uint32
	Flags     uint8
}

const (
	categorical = 1 << 0
	defaultLeft = 1 << 1
	leftLeaf    = 1 << 2
	rightLeaf   = 1 << 3
	missingZero = 1 << 4
	missingNan  = 1 << 5
	catOneHot   = 1 << 6
	catSmall    = 1 << 7
)

// LeavesTree represents a tree in leaves-compatible format
type LeavesTree struct {
	Nodes         []LeavesNode
	LeafValues    []float64
	NCategorical  uint32
	ShrinkageRate float64
	TreeIndex     int
	InternalValue float64 // For init score extraction
}

// Predict makes a prediction using the leaves tree structure
func (t *LeavesTree) Predict(fvals []float64) float64 {
	if len(t.Nodes) == 0 {
		// Constant tree with single leaf value
		// IMPORTANT: LightGBM model files already contain shrinkage-adjusted values
		if len(t.LeafValues) > 0 {
			return t.LeafValues[0]
		}
		return 0.0
	}
	
	idx := uint32(0)
	for {
		node := &t.Nodes[idx]
		left := t.decision(node, fvals[node.Feature])
		
		if left {
			if node.Flags&leftLeaf > 0 {
				// Left child is a leaf
				// IMPORTANT: LightGBM model files already contain shrinkage-adjusted values
				leafValue := t.LeafValues[node.Left]
				return leafValue
			}
			idx = node.Left
		} else {
			if node.Flags&rightLeaf > 0 {
				// Right child is a leaf  
				// IMPORTANT: LightGBM model files already contain shrinkage-adjusted values
				leafValue := t.LeafValues[node.Right]
				return leafValue
			}
			// IMPORTANT: In leaves, right child is at idx+1
			idx++
		}
	}
}

// decision implements the decision logic for a node
func (t *LeavesTree) decision(node *LeavesNode, fval float64) bool {
	if node.Flags&categorical > 0 {
		return t.categoricalDecision(node, fval)
	}
	return t.numericalDecision(node, fval)
}

// numericalDecision implements numerical split decision
func (t *LeavesTree) numericalDecision(node *LeavesNode, fval float64) bool {
	// Handle missing values
	if math.IsNaN(fval) && (node.Flags&missingNan == 0) {
		fval = 0.0
	}
	
	const zeroThreshold = 1e-35
	isZero := fval > -zeroThreshold && fval <= zeroThreshold
	
	if ((node.Flags&missingZero > 0) && isZero) || ((node.Flags&missingNan > 0) && math.IsNaN(fval)) {
		return node.Flags&defaultLeft > 0
	}
	
	// LightGBM uses <= for numerical splits
	return fval <= node.Threshold
}

// categoricalDecision implements categorical split decision
func (t *LeavesTree) categoricalDecision(node *LeavesNode, fval float64) bool {
	// For now, we'll implement basic categorical handling
	// Full implementation would require cat_thresholds and cat_boundaries
	ifval := int32(fval)
	if ifval < 0 {
		return false
	} else if math.IsNaN(fval) {
		if node.Flags&missingNan > 0 {
			return false
		}
		ifval = 0
	}
	
	if node.Flags&catOneHot > 0 {
		return int32(node.Threshold) == ifval
	}
	
	// For simplicity, default to false for other categorical types
	// Full implementation would handle catSmall and regular categorical
	return false
}

// createNumericalNode creates a numerical decision node
func createNumericalNode(feature uint32, missingType uint8, threshold float64, defaultType uint8) LeavesNode {
	node := LeavesNode{}
	node.Feature = feature
	node.Flags = missingType | defaultType
	node.Threshold = threshold
	return node
}