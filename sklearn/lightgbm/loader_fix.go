package lightgbm

import (
	"bufio"
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
	defer file.Close()
	
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