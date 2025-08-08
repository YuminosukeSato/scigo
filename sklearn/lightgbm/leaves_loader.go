package lightgbm

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// LeavesModel represents a LightGBM model in leaves-compatible format
type LeavesModel struct {
	Trees            []LeavesTree
	MaxFeatureIdx    int
	nRawOutputGroups int
	NumClass         int
	NumFeatures      int
	Objective        ObjectiveType
	InitScore        float64
}

// LoadLeavesModelFromFile loads a LightGBM model in leaves-compatible format
func LoadLeavesModelFromFile(filepath string) (*LeavesModel, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()
	
	reader := bufio.NewReader(file)
	return LoadLeavesModelFromReader(reader)
}

// LoadLeavesModelFromReader loads a LightGBM model from a reader
func LoadLeavesModelFromReader(reader *bufio.Reader) (*LeavesModel, error) {
	model := &LeavesModel{
		Trees: []LeavesTree{},
	}
	
	// Read global parameters
	params, err := readParamsUntilBlank(reader)
	if err != nil {
		return nil, err
	}
	
	// Parse global parameters
	if v, ok := params["num_class"]; ok {
		numClass, _ := strconv.Atoi(v)
		model.NumClass = numClass
		model.nRawOutputGroups = numClass
	}
	if model.NumClass == 0 {
		model.NumClass = 1
		model.nRawOutputGroups = 1
	}
	
	if v, ok := params["max_feature_idx"]; ok {
		maxFeature, _ := strconv.Atoi(v)
		model.MaxFeatureIdx = maxFeature
		model.NumFeatures = maxFeature + 1
	}
	
	if v, ok := params["objective"]; ok {
		// Parse objective
		objParts := strings.Fields(v)
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
	}
	
	// Count number of trees from tree_sizes
	treeSizesStr, ok := params["tree_sizes"]
	if !ok {
		return nil, fmt.Errorf("no tree_sizes field")
	}
	treeSizes := strings.Fields(treeSizesStr)
	nTrees := len(treeSizes)
	
	// Read each tree
	for i := 0; i < nTrees; i++ {
		tree, err := readLeavesTree(reader, i)
		if err != nil {
			return nil, fmt.Errorf("error reading tree %d: %w", i, err)
		}
		model.Trees = append(model.Trees, tree)
	}
	
	// Extract InitScore from first tree's internal_value if available
	if len(model.Trees) > 0 && model.Trees[0].InternalValue != 0 {
		model.InitScore = model.Trees[0].InternalValue
	}
	
	// InitScore is not needed for prediction as leaf values already include
	// all necessary adjustments from LightGBM's training process
	// Keep only the original internal_value extraction for compatibility
	
	return model, nil
}

// readLeavesTree reads a single tree in leaves-compatible format
func readLeavesTree(reader *bufio.Reader, treeIndex int) (LeavesTree, error) {
	t := LeavesTree{
		TreeIndex: treeIndex,
	}
	
	params, err := readParamsUntilBlank(reader)
	if err != nil {
		return t, err
	}
	
	// Parse num_leaves
	numLeaves, err := params.toInt("num_leaves")
	if err != nil {
		return t, err
	}
	if numLeaves < 1 {
		return t, fmt.Errorf("num_leaves < 1")
	}
	numNodes := numLeaves - 1
	
	// Parse shrinkage
	if v, ok := params["shrinkage"]; ok {
		shrinkage, _ := strconv.ParseFloat(v, 64)
		t.ShrinkageRate = shrinkage
	}
	
	// Parse leaf values
	leafValues, err := params.toFloat64Slice("leaf_value")
	if err != nil {
		return t, err
	}
	t.LeafValues = leafValues
	
	// Parse internal_value for init score extraction
	if treeIndex == 0 {
		if internalValues, err := params.toFloat64Slice("internal_value"); err == nil && len(internalValues) > 0 {
			t.InternalValue = internalValues[0]
		}
	}
	
	// Special case - constant value tree (single leaf)
	if numLeaves == 1 {
		return t, nil
	}
	
	// Parse tree structure arrays
	leftChilds, err := params.toInt32Slice("left_child")
	if err != nil {
		return t, err
	}
	rightChilds, err := params.toInt32Slice("right_child")
	if err != nil {
		return t, err
	}
	decisionTypes, err := params.toUint32Slice("decision_type")
	if err != nil {
		return t, err
	}
	splitFeatures, err := params.toUint32Slice("split_feature")
	if err != nil {
		return t, err
	}
	thresholds, err := params.toFloat64Slice("threshold")
	if err != nil {
		return t, err
	}
	
	// Create nodes with exact leaves-compatible layout
	createNode := func(idx int32) (LeavesNode, error) {
		node := LeavesNode{}
		
		// Parse missing type from decision_type (bits 2-3)
		missingTypeOrig := (decisionTypes[idx] >> 2) & 3
		missingType := uint8(0)
		if missingTypeOrig == 1 {
			missingType = missingZero
		} else if missingTypeOrig == 2 {
			missingType = missingNan
		}
		
		// Parse default direction (bit 1)
		defaultType := uint8(0)
		if decisionTypes[idx]&(1<<1) > 0 {
			defaultType = defaultLeft
		}
		
		// Check if categorical (bit 0)
		if decisionTypes[idx]&1 > 0 {
			// Categorical node
			node.Flags = categorical | missingType
			node.Feature = splitFeatures[idx]
			node.Threshold = thresholds[idx]
		} else {
			// Numerical node
			node.Flags = missingType | defaultType
			node.Feature = splitFeatures[idx]
			node.Threshold = thresholds[idx]
		}
		
		// Handle children - set leaf flags and indices
		if leftChilds[idx] < 0 {
			node.Flags |= leftLeaf
			// Convert negative child index to leaf value index
			node.Left = uint32(^leftChilds[idx])
		}
		if rightChilds[idx] < 0 {
			node.Flags |= rightLeaf
			// Convert negative child index to leaf value index
			node.Right = uint32(^rightChilds[idx])
		}
		
		return node, nil
	}
	
	// Build tree with exact leaves DFS algorithm matching leaves library
	origNodeIdxStack := make([]uint32, 0, numNodes)
	convNodeIdxStack := make([]uint32, 0, numNodes)
	visited := make([]bool, numNodes)
	t.Nodes = make([]LeavesNode, 0, numNodes)
	
	// Create root node (original index 0)
	node, err := createNode(0)
	if err != nil {
		return t, err
	}
	t.Nodes = append(t.Nodes, node)
	origNodeIdxStack = append(origNodeIdxStack, 0)
	convNodeIdxStack = append(convNodeIdxStack, 0)
	
	
	// Process nodes with right-first DFS to ensure right child is at idx+1
	for len(origNodeIdxStack) > 0 {
		origCurrentIdx := origNodeIdxStack[len(origNodeIdxStack)-1]
		convIdx := convNodeIdxStack[len(convNodeIdxStack)-1]
		
		// First, try to add right child (this ensures right child is at next index)
		if t.Nodes[convIdx].Flags&rightLeaf == 0 {
			origRightIdx := rightChilds[origCurrentIdx]
			if !visited[origRightIdx] {
				node, err := createNode(origRightIdx)
				if err != nil {
					return t, err
				}
				t.Nodes = append(t.Nodes, node)
				convNewIdx := uint32(len(t.Nodes) - 1)
				convNodeIdxStack = append(convNodeIdxStack, convNewIdx)
				origNodeIdxStack = append(origNodeIdxStack, uint32(origRightIdx))
				visited[origRightIdx] = true
				
				
				// Important: Don't set t.Nodes[convIdx].Right here, use implicit idx+1
				continue
			}
		}
		
		// Then, try to add left child
		if t.Nodes[convIdx].Flags&leftLeaf == 0 {
			origLeftIdx := leftChilds[origCurrentIdx]
			if !visited[origLeftIdx] {
				node, err := createNode(origLeftIdx)
				if err != nil {
					return t, err
				}
				t.Nodes = append(t.Nodes, node)
				convNewIdx := uint32(len(t.Nodes) - 1)
				convNodeIdxStack = append(convNodeIdxStack, convNewIdx)
				origNodeIdxStack = append(origNodeIdxStack, uint32(origLeftIdx))
				visited[origLeftIdx] = true
				
				
				// Set explicit left child index
				t.Nodes[convIdx].Left = convNewIdx
				continue
			}
		}
		
		// Pop current node from stack (both children processed or are leaves)
		origNodeIdxStack = origNodeIdxStack[:len(origNodeIdxStack)-1]
		convNodeIdxStack = convNodeIdxStack[:len(convNodeIdxStack)-1]
	}
	
	return t, nil
}


// Helper types and functions
type treeParams map[string]string

func readParamsUntilBlank(reader *bufio.Reader) (treeParams, error) {
	params := make(treeParams)
	emptyLineCount := 0
	
	for {
		line, err := reader.ReadString('\n')
		if err != nil && err != io.EOF {
			return nil, err
		}
		
		line = strings.TrimSpace(line)
		
		// Skip Tree=N lines (these mark section boundaries)
		if strings.HasPrefix(line, "Tree=") {
			continue
		}
		
		// Handle empty lines - allow one empty line, break on consecutive empty lines
		if line == "" {
			emptyLineCount++
			if emptyLineCount >= 2 || (emptyLineCount >= 1 && len(params) > 0) {
				break
			}
			continue
		} else {
			emptyLineCount = 0
		}
		
		// Parse key=value pairs
		if strings.Contains(line, "=") && !strings.HasPrefix(line, "Tree=") {
			parts := strings.SplitN(line, "=", 2)
			if len(parts) == 2 {
				key := strings.TrimSpace(parts[0])
				value := strings.TrimSpace(parts[1])
				params[key] = value
			}
		}
		
		if err == io.EOF {
			break
		}
	}
	
	return params, nil
}

func (p treeParams) toInt(key string) (int, error) {
	v, ok := p[key]
	if !ok {
		return 0, fmt.Errorf("key %s not found", key)
	}
	return strconv.Atoi(v)
}

func (p treeParams) toFloat64Slice(key string) ([]float64, error) {
	v, ok := p[key]
	if !ok {
		return nil, fmt.Errorf("key %s not found", key)
	}
	
	if v == "" {
		return []float64{}, nil
	}
	
	parts := strings.Fields(v)
	result := make([]float64, 0, len(parts))
	for _, part := range parts {
		val, err := strconv.ParseFloat(part, 64)
		if err != nil {
			return nil, err
		}
		result = append(result, val)
	}
	return result, nil
}

func (p treeParams) toInt32Slice(key string) ([]int32, error) {
	v, ok := p[key]
	if !ok {
		return nil, fmt.Errorf("key %s not found", key)
	}
	
	if v == "" {
		return []int32{}, nil
	}
	
	parts := strings.Fields(v)
	result := make([]int32, 0, len(parts))
	for _, part := range parts {
		val, err := strconv.ParseInt(part, 10, 32)
		if err != nil {
			return nil, err
		}
		result = append(result, int32(val))
	}
	return result, nil
}

func (p treeParams) toUint32Slice(key string) ([]uint32, error) {
	v, ok := p[key]
	if !ok {
		return nil, fmt.Errorf("key %s not found", key)
	}
	
	if v == "" {
		return []uint32{}, nil
	}
	
	parts := strings.Fields(v)
	result := make([]uint32, 0, len(parts))
	for _, part := range parts {
		val, err := strconv.ParseUint(part, 10, 32)
		if err != nil {
			return nil, err
		}
		result = append(result, uint32(val))
	}
	return result, nil
}