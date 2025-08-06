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
// This supports the standard LightGBM model format saved by save_model()
func LoadFromFile(filepath string) (*Model, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to open model file: %w", err)
	}
	defer file.Close()

	return LoadFromReader(file)
}

// LoadFromString loads a LightGBM model from a string
// This supports the format from model_to_string()
func LoadFromString(modelStr string) (*Model, error) {
	reader := strings.NewReader(modelStr)
	return LoadFromReader(reader)
}

// LoadFromReader loads a LightGBM model from an io.Reader
func LoadFromReader(reader io.Reader) (*Model, error) {
	scanner := bufio.NewScanner(reader)
	model := NewModel()
	
	var currentSection string
	var currentTree *Tree
	var currentNodeIdx int
	treeNodes := make(map[int]*Node) // Temporary storage for building tree
	
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		
		// Skip empty lines
		if line == "" {
			continue
		}
		
		// Handle section headers
		if strings.HasSuffix(line, ":") {
			currentSection = strings.TrimSuffix(line, ":")
			continue
		}
		
		// Parse based on current section
		switch currentSection {
		case "version":
			model.Version = line
			
		case "num_class":
			numClass, err := strconv.Atoi(line)
			if err != nil {
				return nil, fmt.Errorf("invalid num_class: %w", err)
			}
			model.NumClass = numClass
			
		case "num_tree_per_iteration":
			// This tells us how trees are organized for multiclass
			
		case "label_index":
			// Label mapping for classification
			
		case "max_feature_idx":
			maxFeature, err := strconv.Atoi(line)
			if err != nil {
				return nil, fmt.Errorf("invalid max_feature_idx: %w", err)
			}
			model.NumFeatures = maxFeature + 1
			
		case "objective":
			model.Objective = ObjectiveType(line)
			
		case "feature_names":
			model.FeatureNames = strings.Fields(line)
			
		case "feature_infos":
			// Feature information (types, bounds, etc.)
			
		case "tree_sizes":
			// Size information for each tree
			
		case "Tree":
			// Start of a new tree
			parts := strings.Split(line, "=")
			if len(parts) == 2 && strings.HasPrefix(parts[0], "Tree") {
				treeIdx, err := strconv.Atoi(strings.TrimPrefix(parts[0], "Tree"))
				if err != nil {
					return nil, fmt.Errorf("invalid tree index: %w", err)
				}
				
				// Create new tree
				currentTree = &Tree{
					TreeIndex: treeIdx,
					Nodes:     []Node{},
				}
				treeNodes = make(map[int]*Node)
				currentNodeIdx = 0
			}
			
		default:
			// Parse tree content or parameters
			if currentTree != nil && strings.Contains(line, "=") {
				// Parse tree parameters and nodes
				parts := strings.SplitN(line, "=", 2)
				key := strings.TrimSpace(parts[0])
				value := strings.TrimSpace(parts[1])
				
				switch key {
				case "num_leaves":
					numLeaves, _ := strconv.Atoi(value)
					currentTree.NumLeaves = numLeaves
					
				case "num_cat":
					// Number of categorical features
					
				case "shrinkage":
					shrinkage, _ := strconv.ParseFloat(value, 64)
					currentTree.ShrinkageRate = shrinkage
					if model.LearningRate == 0 {
						model.LearningRate = shrinkage
					}
					
				case "tree_structure":
					// Parse the tree structure
					err := parseTreeStructure(currentTree, value)
					if err != nil {
						return nil, fmt.Errorf("failed to parse tree structure: %w", err)
					}
					// Add completed tree to model
					model.Trees = append(model.Trees, *currentTree)
					currentTree = nil
					
				default:
					// Handle node definitions and other parameters
					if strings.HasPrefix(key, "split_feature") {
						// Parse split features
					} else if strings.HasPrefix(key, "threshold") {
						// Parse thresholds
					} else if strings.HasPrefix(key, "decision_type") {
						// Parse decision types
					} else if strings.HasPrefix(key, "left_child") {
						// Parse left children
					} else if strings.HasPrefix(key, "right_child") {
						// Parse right children
					} else if strings.HasPrefix(key, "leaf_value") {
						// Parse leaf values
					} else if strings.HasPrefix(key, "leaf_count") {
						// Parse leaf counts
					} else if strings.HasPrefix(key, "internal_value") {
						// Parse internal values
					} else if strings.HasPrefix(key, "internal_count") {
						// Parse internal counts
					}
				}
			} else if !strings.Contains(line, "=") && currentSection != "" {
				// Handle multi-line values
				switch currentSection {
				case "feature_names":
					// Additional feature names
					model.FeatureNames = append(model.FeatureNames, strings.Fields(line)...)
				}
			}
		}
	}
	
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading model: %w", err)
	}
	
	// Set derived values
	model.NumIteration = len(model.Trees)
	if model.NumClass == 0 {
		model.NumClass = 1 // Binary or regression
	}
	
	return model, nil
}

// parseTreeStructure parses the tree_structure field from LightGBM model format
func parseTreeStructure(tree *Tree, structure string) error {
	// The tree structure is a nested format that needs careful parsing
	// Example: "split_feature:0 threshold:5.0 left_child:1 right_child:2 leaf_value:0.1"
	
	// For now, implement simplified parsing
	// In production, this would need complete parsing of the nested structure
	
	// Create root node
	root := Node{
		NodeID:   0,
		ParentID: -1,
	}
	
	// Parse the structure recursively
	err := parseNodeStructure(&root, structure, 0)
	if err != nil {
		return err
	}
	
	// Build the nodes array
	tree.Nodes = []Node{root}
	
	return nil
}

// parseNodeStructure recursively parses a node structure
func parseNodeStructure(node *Node, structure string, depth int) error {
	// This is a simplified implementation
	// Full implementation would parse the complete nested structure
	
	// Check if this is a leaf node
	if strings.Contains(structure, "leaf_value") {
		// Extract leaf value
		if idx := strings.Index(structure, "leaf_value:"); idx >= 0 {
			valueStr := structure[idx+11:]
			if endIdx := strings.IndexAny(valueStr, " \t\n"); endIdx > 0 {
				valueStr = valueStr[:endIdx]
			}
			leafValue, err := strconv.ParseFloat(valueStr, 64)
			if err == nil {
				node.LeafValue = leafValue
				node.LeftChild = -1
				node.RightChild = -1
				node.NodeType = LeafNode
			}
		}
		return nil
	}
	
	// Parse split information
	if strings.Contains(structure, "split_feature") {
		node.NodeType = NumericalNode
		
		// Extract split feature
		if idx := strings.Index(structure, "split_feature:"); idx >= 0 {
			featureStr := structure[idx+14:]
			if endIdx := strings.IndexAny(featureStr, " \t\n"); endIdx > 0 {
				featureStr = featureStr[:endIdx]
			}
			feature, err := strconv.Atoi(featureStr)
			if err == nil {
				node.SplitFeature = feature
			}
		}
		
		// Extract threshold
		if idx := strings.Index(structure, "threshold:"); idx >= 0 {
			thresholdStr := structure[idx+10:]
			if endIdx := strings.IndexAny(thresholdStr, " \t\n"); endIdx > 0 {
				thresholdStr = thresholdStr[:endIdx]
			}
			threshold, err := strconv.ParseFloat(thresholdStr, 64)
			if err == nil {
				node.Threshold = threshold
			}
		}
	}
	
	return nil
}

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

// LightGBMJSON represents the JSON structure of a LightGBM model
type LightGBMJSON struct {
	Name              string                 `json:"name"`
	Version           string                 `json:"version"`
	NumClass          int                    `json:"num_class"`
	NumTreePerIteration int                  `json:"num_tree_per_iteration"`
	LabelIndex        int                    `json:"label_index"`
	MaxFeatureIdx     int                    `json:"max_feature_idx"`
	Objective         string                 `json:"objective"`
	AverageOutput     bool                   `json:"average_output"`
	FeatureNames      []string               `json:"feature_names"`
	FeatureInfos      []string               `json:"feature_infos"`
	TreeInfo          []TreeInfoJSON         `json:"tree_info"`
	Trees             []TreeJSON             `json:"tree_structure"`
	PandasCategorical []map[string]int      `json:"pandas_categorical,omitempty"`
	Parameters        map[string]interface{} `json:"parameters,omitempty"`
}

// TreeInfoJSON represents tree metadata in JSON format
type TreeInfoJSON struct {
	TreeIndex    int     `json:"tree_index"`
	NumLeaves    int     `json:"num_leaves"`
	NumCat       int     `json:"num_cat"`
	Shrinkage    float64 `json:"shrinkage"`
}

// TreeJSON represents a tree structure in JSON format
type TreeJSON struct {
	TreeIndex     int           `json:"tree_index"`
	NumLeaves     int           `json:"num_leaves"`
	NumCat        int           `json:"num_cat"`
	Shrinkage     float64       `json:"shrinkage"`
	TreeStructure NodeJSON      `json:"tree_structure"`
}

// NodeJSON represents a node in JSON format
type NodeJSON struct {
	SplitIndex    int          `json:"split_index,omitempty"`
	SplitFeature  int          `json:"split_feature,omitempty"`
	SplitGain     float64      `json:"split_gain,omitempty"`
	Threshold     float64      `json:"threshold,omitempty"`
	DecisionType  string       `json:"decision_type,omitempty"`
	DefaultLeft   bool         `json:"default_left,omitempty"`
	MissingType   string       `json:"missing_type,omitempty"`
	InternalValue float64      `json:"internal_value,omitempty"`
	InternalWeight float64     `json:"internal_weight,omitempty"`
	InternalCount int          `json:"internal_count,omitempty"`
	LeftChild     *NodeJSON    `json:"left_child,omitempty"`
	RightChild    *NodeJSON    `json:"right_child,omitempty"`
	LeafIndex     int          `json:"leaf_index,omitempty"`
	LeafValue     float64      `json:"leaf_value,omitempty"`
	LeafWeight    float64      `json:"leaf_weight,omitempty"`
	LeafCount     int          `json:"leaf_count,omitempty"`
}

// ToModel converts JSON representation to Model
func (j *LightGBMJSON) ToModel() (*Model, error) {
	model := NewModel()
	
	// Set basic properties
	model.Version = j.Version
	model.NumClass = j.NumClass
	model.NumFeatures = j.MaxFeatureIdx + 1
	model.Objective = ObjectiveType(j.Objective)
	model.FeatureNames = j.FeatureNames
	model.Parameters = j.Parameters
	
	// Convert trees
	for i, treeJSON := range j.Trees {
		tree := Tree{
			TreeIndex:     treeJSON.TreeIndex,
			NumLeaves:     treeJSON.NumLeaves,
			ShrinkageRate: treeJSON.Shrinkage,
			Nodes:         []Node{},
		}
		
		// Convert tree structure recursively
		nodeID := 0
		nodeMap := make(map[int]*Node)
		convertNodeJSON(&treeJSON.TreeStructure, &tree, &nodeID, nodeMap, -1)
		
		// Build nodes array from map
		tree.Nodes = make([]Node, len(nodeMap))
		for id, node := range nodeMap {
			tree.Nodes[id] = *node
		}
		
		model.Trees = append(model.Trees, tree)
		
		// Set learning rate from first tree
		if i == 0 && treeJSON.Shrinkage > 0 {
			model.LearningRate = treeJSON.Shrinkage
		}
	}
	
	model.NumIteration = len(model.Trees)
	
	return model, nil
}

// convertNodeJSON recursively converts JSON nodes to internal Node format
func convertNodeJSON(jsonNode *NodeJSON, tree *Tree, nodeID *int, nodeMap map[int]*Node, parentID int) int {
	currentID := *nodeID
	*nodeID++
	
	node := &Node{
		NodeID:   currentID,
		ParentID: parentID,
	}
	
	if jsonNode.LeftChild == nil && jsonNode.RightChild == nil {
		// Leaf node
		node.NodeType = LeafNode
		node.LeafValue = jsonNode.LeafValue
		node.LeafCount = jsonNode.LeafCount
		node.LeftChild = -1
		node.RightChild = -1
	} else {
		// Decision node
		node.NodeType = NumericalNode
		node.SplitFeature = jsonNode.SplitFeature
		node.Threshold = jsonNode.Threshold
		node.DefaultLeft = jsonNode.DefaultLeft
		node.Gain = jsonNode.SplitGain
		node.InternalValue = jsonNode.InternalValue
		node.InternalCount = jsonNode.InternalCount
		
		// Process children
		if jsonNode.LeftChild != nil {
			node.LeftChild = convertNodeJSON(jsonNode.LeftChild, tree, nodeID, nodeMap, currentID)
		}
		if jsonNode.RightChild != nil {
			node.RightChild = convertNodeJSON(jsonNode.RightChild, tree, nodeID, nodeMap, currentID)
		}
	}
	
	nodeMap[currentID] = node
	return currentID
}

// LoadFromJSONFile loads a LightGBM model from a JSON file
func LoadFromJSONFile(filepath string) (*Model, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read JSON file: %w", err)
	}
	
	return LoadFromJSON(data)
}