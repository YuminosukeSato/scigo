#!/usr/bin/env python3
"""
LightGBMの特定ツリーの詳細分析
"""

import lightgbm as lgb
import numpy as np

def trace_tree_prediction(model, tree_idx, sample_features):
    """特定のツリーで予測をトレースする"""
    
    model_dump = model.dump_model()
    tree_info = model_dump['tree_info'][tree_idx]
    tree_struct = tree_info['tree_structure']
    
    print(f"=== Tree {tree_idx} 詳細トレース ===")
    print(f"Shrinkage: {tree_info['shrinkage']}")
    print(f"Num leaves: {tree_info['num_leaves']}")
    print(f"Features: {sample_features}")
    print()
    
    def traverse_node(node, depth=0):
        indent = "  " * depth
        
        if 'leaf_value' in node:
            # リーフノード
            leaf_val = node['leaf_value']
            shrinkage = tree_info['shrinkage']
            final_val = leaf_val * shrinkage
            print(f"{indent}LEAF: raw_value={leaf_val:.6f}, shrinkage={shrinkage}, final={final_val:.6f}")
            return final_val
        
        # 内部ノード
        feature_idx = node['split_feature']
        threshold = node['threshold']
        feature_val = sample_features[feature_idx]
        
        print(f"{indent}NODE: feature[{feature_idx}]={feature_val:.6f} vs threshold={threshold:.6f}")
        
        if feature_val <= threshold:
            print(f"{indent}-> Going LEFT")
            return traverse_node(node['left_child'], depth + 1)
        else:
            print(f"{indent}-> Going RIGHT")
            return traverse_node(node['right_child'], depth + 1)
    
    predicted_value = traverse_node(tree_struct)
    
    # LightGBMの実際の予測と比較
    actual_pred = model.predict(np.array([sample_features]), start_iteration=tree_idx, num_iteration=1)[0]
    
    print(f"\nトレース結果: {predicted_value:.6f}")
    print(f"LightGBM実際: {actual_pred:.6f}")
    print(f"差: {abs(predicted_value - actual_pred):.6f}")
    
    return predicted_value, actual_pred

def main():
    # モデルとサンプルデータを読み込み
    model = lgb.Booster(model_file='testdata/compatibility/regression_model.txt')
    sample = [1.59040357, -0.39398668, 0.04092475, -0.99844085, 1.99137042, 
              0.43494104, 1.62325669, -0.5691482, -0.79711357, 0.39291351]
    
    # Tree 1 (問題のあるツリー) を詳細分析
    print("Tree 1 詳細分析:")
    trace_tree_prediction(model, 1, sample)
    
    print("\n" + "="*50 + "\n")
    
    # Tree 2も確認
    print("Tree 2 詳細分析:")
    trace_tree_prediction(model, 2, sample)

if __name__ == "__main__":
    main()