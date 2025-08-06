package lightgbm

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// LoadFromFile はテキストファイルからLightGBMモデルを読み込みます
func LoadFromFile(filepath string) (*Model, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()
	
	return LoadFromReader(file)
}

// LoadFromString は文字列形式からLightGBMモデルを読み込みます
func LoadFromString(modelStr string) (*Model, error) {
	reader := strings.NewReader(modelStr)
	return LoadFromReader(reader)
}

// LoadFromReader はio.ReaderからLightGBMモデルを読み込みます
func LoadFromReader(reader io.Reader) (*Model, error) {
	scanner := bufio.NewScanner(reader)
	model := NewModel()
	
	var currentTree *Tree
	inTree := false
	treeParams := make(map[string]string)
	
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		
		// 空行をスキップ
		if line == "" {
			// ツリー処理中に空行に到達した場合、ツリーを完成させる
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
		
		// ツリーヘッダーかどうかを確認
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
		
		// key=valueペアをパース
		if strings.Contains(line, "=") {
			parts := strings.SplitN(line, "=", 2)
			key := strings.TrimSpace(parts[0])
			value := strings.TrimSpace(parts[1])
			
			if inTree {
				// ツリーパラメータを保存
				treeParams[key] = value
			} else {
				// モデルパラメータをパース
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
					// 目的関数をパース、例: "binary sigmoid:1"
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
	
	// 最後のツリーが存在する場合は処理
	if inTree && currentTree != nil {
		if err := finalizeTree(currentTree, treeParams); err != nil {
			return nil, err
		}
		model.Trees = append(model.Trees, *currentTree)
	}
	
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading model: %w", err)
	}
	
	// 派生値を設定
	model.NumIteration = len(model.Trees)
	if model.NumClass == 0 {
		model.NumClass = 1
	}
	
	return model, nil
}

// finalizeTree はツリーパラメータをパースしてツリーノードを構築します
func finalizeTree(tree *Tree, params map[string]string) error {
	// num_leavesをパース
	if v, ok := params["num_leaves"]; ok {
		numLeaves, _ := strconv.Atoi(v)
		tree.NumLeaves = numLeaves
	}
	
	// shrinkageをパース
	if v, ok := params["shrinkage"]; ok {
		shrinkage, _ := strconv.ParseFloat(v, 64)
		tree.ShrinkageRate = shrinkage
	}
	
	// 値の配列をパース
	splitFeatures := parseIntArray(params["split_feature"])
	thresholds := parseFloatArray(params["threshold"])
	leftChildren := parseIntArray(params["left_child"])
	rightChildren := parseIntArray(params["right_child"])
	leafValues := parseFloatArray(params["leaf_value"])
	
	// ノードを構築
	numInternalNodes := len(splitFeatures)
	numLeaves := len(leafValues)
	totalNodes := numInternalNodes + numLeaves
	
	tree.Nodes = make([]Node, 0, totalNodes)
	
	// 内部ノードを作成
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
		
		// 実際にリーフかどうかを確認（-1はリーフを意味する）
		if leftChildren[i] < 0 && rightChildren[i] < 0 {
			// これは実際にリーフノードなので、値を取得
			leafIdx := -(leftChildren[i] + 1)
			if leafIdx < len(leafValues) {
				node.LeafValue = leafValues[leafIdx]
				node.NodeType = LeafNode
			}
		}
		
		tree.Nodes = append(tree.Nodes, node)
	}
	
	// まだ追加されていないリーフノードを追加
	leafNodeID := numInternalNodes
	for i := 0; i < numLeaves; i++ {
		// このリーフがすでに内部ノードの一部として追加されているか確認
		alreadyAdded := false
		for j := 0; j < numInternalNodes; j++ {
			if leftChildren[j] == -(i+1) || rightChildren[j] == -(i+1) {
				// このリーフは参照されているが、適切に作成する必要がある
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
			// 独立したリーフノード
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

// parseIntArray はスペース区切りの整数文字列をパースします
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

// parseFloatArray はスペース区切りの浮動小数点文字列をパースします
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