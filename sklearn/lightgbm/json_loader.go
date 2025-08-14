package lightgbm

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// JSONModel represents the top-level structure of a LightGBM JSON model
type JSONModel struct {
	Name                string                 `json:"name"`
	Version             string                 `json:"version"`
	NumClass            int                    `json:"num_class"`
	NumTreePerIteration int                    `json:"num_tree_per_iteration"`
	LabelIndex          int                    `json:"label_index"`
	MaxFeatureIdx       int                    `json:"max_feature_idx"`
	Objective           string                 `json:"objective"`
	AverageOutput       bool                   `json:"average_output"`
	FeatureNames        []string               `json:"feature_names"`
	MonotoneConstraints []int                  `json:"monotone_constraints"`
	FeatureInfos        map[string]FeatureInfo `json:"feature_infos"`
	TreeInfo            []JSONTreeInfo         `json:"tree_info"`
	PandasCategorical   []interface{}          `json:"pandas_categorical,omitempty"`
}

// FeatureInfo contains min/max values and categorical values for a feature
type FeatureInfo struct {
	MinValue float64   `json:"min_value"`
	MaxValue float64   `json:"max_value"`
	Values   []float64 `json:"values"`
}

// JSONTreeInfo represents information about a single tree
type JSONTreeInfo struct {
	TreeIndex     int          `json:"tree_index"`
	NumLeaves     int          `json:"num_leaves"`
	NumCat        int          `json:"num_cat"`
	Shrinkage     float64      `json:"shrinkage"`
	TreeStructure JSONTreeNode `json:"tree_structure"`
}

// JSONTreeNode represents a node in the tree (can be internal or leaf)
type JSONTreeNode struct {
	// Internal node fields
	SplitIndex     int           `json:"split_index,omitempty"`
	SplitFeature   int           `json:"split_feature,omitempty"`
	SplitGain      float64       `json:"split_gain,omitempty"`
	Threshold      interface{}   `json:"threshold,omitempty"` // Can be float64 or string for categorical
	DecisionType   string        `json:"decision_type,omitempty"`
	DefaultLeft    bool          `json:"default_left,omitempty"`
	MissingType    string        `json:"missing_type,omitempty"`
	InternalValue  float64       `json:"internal_value,omitempty"`
	InternalWeight float64       `json:"internal_weight,omitempty"`
	InternalCount  int           `json:"internal_count,omitempty"`
	LeftChild      *JSONTreeNode `json:"left_child,omitempty"`
	RightChild     *JSONTreeNode `json:"right_child,omitempty"`

	// Categorical split fields
	SplitIndices []int `json:"split_indices,omitempty"`

	// Leaf node fields
	LeafIndex  int     `json:"leaf_index,omitempty"`
	LeafValue  float64 `json:"leaf_value,omitempty"`
	LeafWeight float64 `json:"leaf_weight,omitempty"`
	LeafCount  int     `json:"leaf_count,omitempty"`
}

// LoadJSONModelFromFile loads a LightGBM model from JSON file
func LoadJSONModelFromFile(filePath string) (*LeavesModel, error) {
	// Validate file path
	cleanPath := filepath.Clean(filePath)
	if strings.Contains(cleanPath, "..") {
		return nil, fmt.Errorf("path traversal detected in file path: %s", filePath)
	}

	// Read file
	data, err := os.ReadFile(cleanPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	// Parse JSON
	var jsonModel JSONModel
	if err := json.Unmarshal(data, &jsonModel); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	// Convert to LeavesModel
	return convertJSONToLeavesModel(&jsonModel)
}

// convertJSONToLeavesModel converts JSON model to LeavesModel format
func convertJSONToLeavesModel(jsonModel *JSONModel) (*LeavesModel, error) {
	model := &LeavesModel{
		Trees:            make([]LeavesTree, 0, len(jsonModel.TreeInfo)),
		MaxFeatureIdx:    jsonModel.MaxFeatureIdx,
		nRawOutputGroups: jsonModel.NumTreePerIteration,
		NumClass:         jsonModel.NumClass,
		NumFeatures:      jsonModel.MaxFeatureIdx + 1,
	}

	// Parse objective
	objective := parseObjective(jsonModel.Objective)
	model.Objective = objective

	// Convert each tree
	for _, treeInfo := range jsonModel.TreeInfo {
		tree, err := convertJSONTree(&treeInfo)
		if err != nil {
			return nil, fmt.Errorf("failed to convert tree %d: %w", treeInfo.TreeIndex, err)
		}
		model.Trees = append(model.Trees, tree)
	}

	return model, nil
}

// parseObjective parses the objective string and returns ObjectiveType
func parseObjective(obj string) ObjectiveType {
	// Remove parameters from objective string
	parts := strings.Fields(obj)
	if len(parts) > 0 {
		switch parts[0] {
		case "binary":
			return BinaryLogistic
		case "regression":
			return RegressionL2
		case "multiclass":
			return MulticlassSoftmax
		case "multiclass_logloss":
			return MulticlassLogLoss
		default:
			return ObjectiveType(parts[0])
		}
	}
	return RegressionL2
}

// convertJSONTree converts a JSON tree structure to LeavesTree
func convertJSONTree(treeInfo *JSONTreeInfo) (LeavesTree, error) {
	tree := LeavesTree{
		TreeIndex:     treeInfo.TreeIndex,
		ShrinkageRate: treeInfo.Shrinkage,
	}

	// Collect all leaf values and build nodes
	leafValues := make([]float64, 0)
	nodes := make([]LeavesNode, 0)

	// If the tree is a single leaf (no splits)
	if treeInfo.TreeStructure.LeafIndex >= 0 {
		leafValues = append(leafValues, treeInfo.TreeStructure.LeafValue)
		tree.LeafValues = leafValues
		tree.Nodes = nodes
		return tree, nil
	}

	// Build the tree structure using DFS
	var leafIndex uint32 = 0

	var buildNodes func(jsonNode *JSONTreeNode) uint32
	buildNodes = func(jsonNode *JSONTreeNode) uint32 {
		if jsonNode == nil {
			return 0
		}

		// Check if this is a leaf node
		if jsonNode.LeftChild == nil && jsonNode.RightChild == nil {
			// This is a leaf
			leafValues = append(leafValues, jsonNode.LeafValue)
			idx := leafIndex
			leafIndex++
			return idx | (1 << 31) // Mark as leaf with high bit
		}

		// This is an internal node
		nodeIdx := uint32(len(nodes))
		node := LeavesNode{
			Feature: uint32(jsonNode.SplitFeature),
		}

		// Parse threshold value (can be float64 or string for categorical)
		switch v := jsonNode.Threshold.(type) {
		case float64:
			node.Threshold = v
		case string:
			// For categorical splits, threshold might be a string
			// Try to parse it as a number
			var val float64
			if _, err := fmt.Sscanf(v, "%f", &val); err == nil {
				node.Threshold = val
			}
		default:
			node.Threshold = 0.0
		}

		// Set flags based on decision type and missing type
		if jsonNode.DecisionType == "==" {
			// Categorical split
			node.Flags |= categorical
			// Store split indices if available
			if len(jsonNode.SplitIndices) > 0 {
				// Note: We would need to extend LeavesNode to store categories
				// For now, we'll use threshold to encode the first category
				node.Threshold = float64(jsonNode.SplitIndices[0])
			}
		}

		// Handle missing values
		switch jsonNode.MissingType {
		case "Zero":
			node.Flags |= missingZero
		case "NaN":
			node.Flags |= missingNan
		}

		// Set default direction
		if jsonNode.DefaultLeft {
			node.Flags |= defaultLeft
		}

		// Add node first to get its index
		nodes = append(nodes, node)

		// Process children
		leftResult := buildNodes(jsonNode.LeftChild)
		rightResult := buildNodes(jsonNode.RightChild)

		// Update node with children info
		if leftResult&(1<<31) != 0 {
			// Left child is a leaf
			nodes[nodeIdx].Flags |= leftLeaf
			nodes[nodeIdx].Left = leftResult & 0x7FFFFFFF
		} else {
			nodes[nodeIdx].Left = leftResult
		}

		if rightResult&(1<<31) != 0 {
			// Right child is a leaf
			nodes[nodeIdx].Flags |= rightLeaf
			nodes[nodeIdx].Right = rightResult & 0x7FFFFFFF
		} else {
			// For leaves format, right child is implicitly at nodeIdx+1
			// We need to rearrange nodes to ensure this
			nodes[nodeIdx].Right = nodeIdx + 1
		}

		return nodeIdx
	}

	// Build the tree starting from root
	buildNodes(&treeInfo.TreeStructure)

	tree.Nodes = nodes
	tree.LeafValues = leafValues

	return tree, nil
}

// IsJSONModel checks if the file is a JSON model by attempting to parse it
func IsJSONModel(filePath string) bool {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return false
	}

	var jsonModel JSONModel
	return json.Unmarshal(data, &jsonModel) == nil && jsonModel.Version != ""
}

// LoadModelAutoDetect automatically detects the model format and loads it
func LoadModelAutoDetect(filePath string) (*LeavesModel, error) {
	// Check if it's JSON format
	if IsJSONModel(filePath) {
		return LoadJSONModelFromFile(filePath)
	}

	// Otherwise, assume it's text format
	return LoadLeavesModelFromFile(filePath)
}
