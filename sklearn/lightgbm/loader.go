package lightgbm

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// LoadFromFile loads a LightGBM model from a text file
func LoadFromFile(filepath string) (*Model, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer func() { _ = file.Close() }()

	return LoadFromReader(file)
}

// LoadFromString loads a LightGBM model from string format
func LoadFromString(modelStr string) (*Model, error) {
	reader := strings.NewReader(modelStr)
	return LoadFromReader(reader)
}

// LoadFromReader loads a LightGBM model from an io.Reader
func LoadFromReader(reader io.Reader) (*Model, error) {
	scanner := bufio.NewScanner(reader)
	model := NewModel()

	var currentTree *Tree
	inTree := false
	treeParams := make(map[string]string)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		// Skip empty lines
		if line == "" {
			// If we were in a tree and hit empty line, finalize the tree
			if inTree && currentTree != nil {
				if err := finalizeTree(currentTree, treeParams); err != nil {
					return nil, err
				}
				model.Trees = append(model.Trees, *currentTree)
				currentTree = nil
				inTree = false
				treeParams = make(map[string]string)
			}
			continue
		}

		// Check if this is a tree header
		if strings.HasPrefix(line, "Tree=") {
			parts := strings.Split(line, "=")
			if len(parts) == 2 {
				treeIdx, err := strconv.Atoi(parts[1])
				if err != nil {
					return nil, fmt.Errorf("invalid tree index: %w", err)
				}
				currentTree = &Tree{
					TreeIndex: treeIdx,
					Nodes:     []Node{},
				}
				inTree = true
				treeParams = make(map[string]string)
			}
			continue
		}

		// Parse key=value pairs
		if strings.Contains(line, "=") {
			parts := strings.SplitN(line, "=", 2)
			key := strings.TrimSpace(parts[0])
			value := strings.TrimSpace(parts[1])

			if inTree {
				// Store tree parameters
				treeParams[key] = value
			} else {
				// Parse model parameters
				switch key {
				case "version":
					model.Version = value
				case "num_class":
					numClass, _ := strconv.Atoi(value)
					model.NumClass = numClass
				case "max_feature_idx":
					maxFeature, _ := strconv.Atoi(value)
					model.NumFeatures = maxFeature + 1
				case "objective":
					// Parse objective, e.g., "binary sigmoid:1"
					objParts := strings.Fields(value)
					if len(objParts) > 0 {
						switch objParts[0] {
						case "binary":
							model.Objective = BinaryLogistic
						case "regression":
							model.Objective = RegressionL2
						case "multiclass":
							model.Objective = MulticlassSoftmax
						default:
							model.Objective = ObjectiveType(objParts[0])
						}
					}
				case "feature_names":
					model.FeatureNames = strings.Fields(value)
				}
			}
		}
	}

	// Handle last tree if exists
	if inTree && currentTree != nil {
		if err := finalizeTree(currentTree, treeParams); err != nil {
			return nil, err
		}
		model.Trees = append(model.Trees, *currentTree)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading model: %w", err)
	}

	// Set derived values
	model.NumIteration = len(model.Trees)
	if model.NumClass == 0 {
		model.NumClass = 1
	}

	return model, nil
}

// finalizeTree parses the tree parameters and constructs the tree nodes
func finalizeTree(tree *Tree, params map[string]string) error {
	// Parse tree parameters for future expansion
	_ = params // Currently unused, reserved for future enhancements
	// Parse num_leaves
	if v, ok := params["num_leaves"]; ok {
		numLeaves, _ := strconv.Atoi(v)
		tree.NumLeaves = numLeaves
	}

	// Parse shrinkage
	if v, ok := params["shrinkage"]; ok {
		shrinkage, _ := strconv.ParseFloat(v, 64)
		tree.ShrinkageRate = shrinkage
	}

	// Parse arrays of values
	splitFeatures := parseIntArray(params["split_feature"])
	thresholds := parseFloatArray(params["threshold"])
	leftChildren := parseIntArray(params["left_child"])
	rightChildren := parseIntArray(params["right_child"])
	leafValues := parseFloatArray(params["leaf_value"])

	// Build nodes
	numInternalNodes := len(splitFeatures)
	numLeaves := len(leafValues)
	totalNodes := numInternalNodes + numLeaves

	tree.Nodes = make([]Node, 0, totalNodes)

	// Create internal nodes
	for i := 0; i < numInternalNodes; i++ {
		node := Node{
			NodeID:       i,
			ParentID:     -1, // Will be set later
			LeftChild:    leftChildren[i],
			RightChild:   rightChildren[i],
			SplitFeature: splitFeatures[i],
			Threshold:    thresholds[i],
			NodeType:     NumericalNode,
		}

		// Check if it's actually a leaf (-1 means leaf)
		if leftChildren[i] < 0 && rightChildren[i] < 0 {
			// This is actually a leaf node, get its value
			leafIdx := -(leftChildren[i] + 1)
			if leafIdx < len(leafValues) {
				node.LeafValue = leafValues[leafIdx]
				node.NodeType = LeafNode
			}
		}

		tree.Nodes = append(tree.Nodes, node)
	}

	// Add leaf nodes that aren't already added
	leafNodeID := numInternalNodes
	for i := 0; i < numLeaves; i++ {
		// Check if this leaf was already added as part of internal nodes
		alreadyAdded := false
		for j := 0; j < numInternalNodes; j++ {
			if leftChildren[j] == -(i+1) || rightChildren[j] == -(i+1) {
				// This leaf is referenced, but we need to create it properly
				node := Node{
					NodeID:     leafNodeID,
					ParentID:   j,
					LeftChild:  -1,
					RightChild: -1,
					LeafValue:  leafValues[i],
					NodeType:   LeafNode,
				}
				tree.Nodes = append(tree.Nodes, node)
				leafNodeID++
				alreadyAdded = true
				break
			}
		}

		if !alreadyAdded {
			// Standalone leaf node
			node := Node{
				NodeID:     leafNodeID,
				ParentID:   -1,
				LeftChild:  -1,
				RightChild: -1,
				LeafValue:  leafValues[i],
				NodeType:   LeafNode,
			}
			tree.Nodes = append(tree.Nodes, node)
			leafNodeID++
		}
	}

	return nil
}

// parseIntArray parses a space-separated string of integers
func parseIntArray(s string) []int {
	if s == "" {
		return []int{}
	}
	parts := strings.Fields(s)
	result := make([]int, 0, len(parts))
	for _, p := range parts {
		if v, err := strconv.Atoi(p); err == nil {
			result = append(result, v)
		}
	}
	return result
}

// parseFloatArray parses a space-separated string of floats
func parseFloatArray(s string) []float64 {
	if s == "" {
		return []float64{}
	}
	parts := strings.Fields(s)
	result := make([]float64, 0, len(parts))
	for _, p := range parts {
		if v, err := strconv.ParseFloat(p, 64); err == nil {
			result = append(result, v)
		}
	}
	return result
}

// parseTreeStructure is kept for compatibility but not really used in text format
// Deprecated: This function is no longer used and will be removed in future versions
// func parseTreeStructure(tree *Tree, structure string) error {
// 	// This function was for JSON-style tree structures
// 	// The text format uses separate arrays for each parameter
// 	return nil
// }

// LoadFromJSON loads a LightGBM model from JSON format
// This supports the format from dump_model()
func LoadFromJSON(jsonData []byte) (*Model, error) {
	var jsonModel LightGBMJSON
	err := json.Unmarshal(jsonData, &jsonModel)
	if err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	return jsonModel.ToModel()
}

// LoadFromJSONFile loads a LightGBM model from a JSON file
func LoadFromJSONFile(filepath string) (*Model, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read JSON file: %w", err)
	}

	return LoadFromJSON(data)
}

// LightGBMJSON represents the JSON structure of a LightGBM model
type LightGBMJSON struct {
	Name                string              `json:"name"`
	Version             string              `json:"version"`
	NumClass            int                 `json:"num_class"`
	NumTreePerIteration int                 `json:"num_tree_per_iteration"`
	LabelIndex          int                 `json:"label_index"`
	MaxFeatureIdx       int                 `json:"max_feature_idx"`
	Objective           string              `json:"objective"`
	FeatureNames        []string            `json:"feature_names"`
	FeatureInfos        []string            `json:"feature_infos"`
	TreeInfo            []TreeInfoJSON      `json:"tree_info"`
	TreeStructure       []TreeStructureJSON `json:"tree_structure"`
}

// TreeInfoJSON represents tree metadata in JSON format
type TreeInfoJSON struct {
	TreeIndex int     `json:"tree_index"`
	NumLeaves int     `json:"num_leaves"`
	NumCat    int     `json:"num_cat"`
	Shrinkage float64 `json:"shrinkage"`
}

// TreeStructureJSON represents a tree structure in JSON format
type TreeStructureJSON struct {
	TreeIndex     int         `json:"tree_index"`
	NumLeaves     int         `json:"num_leaves"`
	NumCat        int         `json:"num_cat"`
	Shrinkage     float64     `json:"shrinkage"`
	TreeStructure interface{} `json:"tree_structure"`
}

// NodeJSON represents a node in the JSON tree structure
type NodeJSON struct {
	SplitIndex     int       `json:"split_index,omitempty"`
	SplitFeature   int       `json:"split_feature,omitempty"`
	SplitGain      float64   `json:"split_gain,omitempty"`
	Threshold      float64   `json:"threshold,omitempty"`
	DecisionType   string    `json:"decision_type,omitempty"`
	DefaultLeft    bool      `json:"default_left,omitempty"`
	MissingType    string    `json:"missing_type,omitempty"`
	InternalValue  float64   `json:"internal_value,omitempty"`
	InternalWeight float64   `json:"internal_weight,omitempty"`
	InternalCount  int       `json:"internal_count,omitempty"`
	LeftChild      *NodeJSON `json:"left_child,omitempty"`
	RightChild     *NodeJSON `json:"right_child,omitempty"`
	LeafIndex      int       `json:"leaf_index,omitempty"`
	LeafValue      float64   `json:"leaf_value,omitempty"`
	LeafWeight     float64   `json:"leaf_weight,omitempty"`
	LeafCount      int       `json:"leaf_count,omitempty"`
}

// ToModel converts a LightGBMJSON to our Model structure
func (lgbJSON *LightGBMJSON) ToModel() (*Model, error) {
	model := NewModel()
	model.Version = lgbJSON.Version
	model.NumClass = lgbJSON.NumClass
	model.NumFeatures = lgbJSON.MaxFeatureIdx + 1
	model.FeatureNames = lgbJSON.FeatureNames

	// Parse objective
	switch lgbJSON.Objective {
	case "binary", "binary sigmoid", "binary_logloss":
		model.Objective = BinaryLogistic
	case "regression", "regression_l2", "l2", "mean_squared_error":
		model.Objective = RegressionL2
	case "regression_l1", "l1", "mean_absolute_error":
		model.Objective = RegressionL1
	case "multiclass", "softmax", "multiclassova":
		model.Objective = MulticlassSoftmax
	default:
		model.Objective = ObjectiveType(lgbJSON.Objective)
	}

	// Parse trees - combine TreeInfo and TreeStructure
	for i, treeInfo := range lgbJSON.TreeInfo {
		tree := Tree{
			TreeIndex:     treeInfo.TreeIndex,
			NumLeaves:     treeInfo.NumLeaves,
			ShrinkageRate: treeInfo.Shrinkage,
			Nodes:         []Node{},
		}

		// Parse tree structure if available
		if i < len(lgbJSON.TreeStructure) {
			treeStruct := lgbJSON.TreeStructure[i]
			if treeStructMap, ok := treeStruct.TreeStructure.(map[string]interface{}); ok {
				rootNode := parseNodeFromJSON(treeStructMap)
				flattenTree(&tree, rootNode, -1)
			} else if nodeJSON, ok := treeStruct.TreeStructure.(*NodeJSON); ok {
				// Direct NodeJSON structure
				flattenTree(&tree, nodeJSON, -1)
			}
		}

		// Update tree metadata
		tree.NumNodes = len(tree.Nodes)
		if tree.NumLeaves == 0 {
			// Count leaf nodes
			leafCount := 0
			for _, node := range tree.Nodes {
				if node.NodeType == LeafNode {
					leafCount++
				}
			}
			tree.NumLeaves = leafCount
		}

		model.Trees = append(model.Trees, tree)
	}

	model.NumIteration = len(model.Trees)
	return model, nil
}

// parseNodeFromJSON converts a JSON node to our NodeJSON structure
func parseNodeFromJSON(nodeMap map[string]interface{}) *NodeJSON {
	// Use json.Marshal and Unmarshal to map the generic interface to a struct
	jsonBytes, _ := json.Marshal(nodeMap)
	var nodeJSON NodeJSON
	json.Unmarshal(jsonBytes, &nodeJSON)
	return &nodeJSON
}

// flattenTree converts a tree structure to a flat array of nodes
func flattenTree(tree *Tree, nodeJSON *NodeJSON, parentID int) int {
	node := &Node{
		ParentID: parentID,
	}

	// Check if it's a leaf node
	if nodeJSON.LeftChild == nil && nodeJSON.RightChild == nil {
		node.NodeType = LeafNode
		node.LeafValue = nodeJSON.LeafValue
		node.LeafCount = nodeJSON.LeafCount
		node.LeftChild = -1
		node.RightChild = -1
		nodeID := len(tree.Nodes)
		node.NodeID = nodeID
		tree.Nodes = append(tree.Nodes, *node)
		return nodeID
	}

	// It's an internal node
	node.NodeType = NumericalNode
	node.SplitFeature = nodeJSON.SplitFeature
	node.Threshold = nodeJSON.Threshold
	node.Gain = nodeJSON.SplitGain
	node.DefaultLeft = nodeJSON.DefaultLeft
	node.InternalValue = nodeJSON.InternalValue
	node.InternalCount = nodeJSON.InternalCount

	nodeID := len(tree.Nodes)
	node.NodeID = nodeID
	tree.Nodes = append(tree.Nodes, *node) // Add placeholder for current node

	// Recursively process children
	if nodeJSON.LeftChild != nil {
		leftID := flattenTree(tree, nodeJSON.LeftChild, nodeID)
		tree.Nodes[nodeID].LeftChild = leftID
	} else {
		tree.Nodes[nodeID].LeftChild = -1
	}

	if nodeJSON.RightChild != nil {
		rightID := flattenTree(tree, nodeJSON.RightChild, nodeID)
		tree.Nodes[nodeID].RightChild = rightID
	} else {
		tree.Nodes[nodeID].RightChild = -1
	}

	return nodeID
}

// SaveToJSON saves the model to a JSON file
func (m *Model) SaveToJSON(filepath string) error {
	jsonData, err := m.ToJSON()
	if err != nil {
		return err
	}
	return os.WriteFile(filepath, jsonData, 0644)
}

// SaveToJSONString returns the model as a JSON string
func (m *Model) SaveToJSONString() (string, error) {
	jsonData, err := m.ToJSON()
	if err != nil {
		return "", err
	}
	return string(jsonData), nil
}

// ToJSON converts the model to JSON format
func (m *Model) ToJSON() ([]byte, error) {
	jsonModel := m.toLightGBMJSON()
	return json.MarshalIndent(jsonModel, "", "  ")
}

// toLightGBMJSON converts the model to LightGBMJSON structure
func (m *Model) toLightGBMJSON() *LightGBMJSON {
	jsonModel := &LightGBMJSON{
		Name:                "tree",
		Version:             "v3",
		NumClass:            m.NumClass,
		NumTreePerIteration: 1,
		LabelIndex:          0,
		MaxFeatureIdx:       m.NumFeatures - 1,
		Objective:           string(m.Objective),
		FeatureNames:        m.FeatureNames,
		FeatureInfos:        make([]string, m.NumFeatures),
		TreeInfo:            make([]TreeInfoJSON, len(m.Trees)),
		TreeStructure:       make([]TreeStructureJSON, len(m.Trees)),
	}

	// Convert each tree
	for i, tree := range m.Trees {
		jsonModel.TreeInfo[i] = TreeInfoJSON{
			TreeIndex: i,
			NumLeaves: tree.NumLeaves,
			NumCat:    0,
			Shrinkage: tree.ShrinkageRate,
		}

		// Convert tree structure
		if len(tree.Nodes) > 0 {
			jsonModel.TreeStructure[i] = TreeStructureJSON{
				TreeIndex:     i,
				TreeStructure: treeToNodeJSON(&tree, 0),
			}
		}
	}

	return jsonModel
}

// treeToNodeJSON converts a tree node to NodeJSON recursively
func treeToNodeJSON(tree *Tree, nodeID int) *NodeJSON {
	if nodeID < 0 || nodeID >= len(tree.Nodes) {
		return nil
	}

	node := &tree.Nodes[nodeID]

	if node.NodeType == LeafNode {
		return &NodeJSON{
			LeafIndex: nodeID,
			LeafValue: node.LeafValue,
			LeafCount: node.LeafCount,
		}
	}

	// Internal node
	nodeJSON := &NodeJSON{
		SplitIndex:    nodeID,
		SplitFeature:  node.SplitFeature,
		SplitGain:     node.Gain,
		Threshold:     node.Threshold,
		DecisionType:  "<=",
		DefaultLeft:   node.DefaultLeft,
		MissingType:   "None",
		InternalValue: node.InternalValue,
		InternalCount: node.InternalCount,
	}

	// Add children recursively
	if node.LeftChild >= 0 {
		nodeJSON.LeftChild = treeToNodeJSON(tree, node.LeftChild)
	}
	if node.RightChild >= 0 {
		nodeJSON.RightChild = treeToNodeJSON(tree, node.RightChild)
	}

	return nodeJSON
}
