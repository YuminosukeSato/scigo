package lightgbm

import (
	"fmt"
	"strings"
)

// ParameterMapper handles mapping between Python LightGBM parameters and Go implementation
type ParameterMapper struct {
	// Mapping from Python parameter names to Go field names
	pythonToGo map[string]string
	// Mapping from Go field names to Python parameter names
	goToPython map[string]string
	// Parameter aliases (multiple Python names for same parameter)
	aliases map[string]string
	// Default values for parameters
	defaults map[string]interface{}
}

// NewParameterMapper creates a new parameter mapper with LightGBM compatible mappings
func NewParameterMapper() *ParameterMapper {
	pm := &ParameterMapper{
		pythonToGo: make(map[string]string),
		goToPython: make(map[string]string),
		aliases:    make(map[string]string),
		defaults:   make(map[string]interface{}),
	}

	pm.initializeMappings()
	pm.initializeDefaults()

	return pm
}

// initializeMappings sets up all parameter mappings
func (pm *ParameterMapper) initializeMappings() {
	// Core Parameters
	pm.addMapping("n_estimators", "NumIterations", []string{"num_iterations", "num_iteration", "num_tree", "num_trees", "num_round", "num_rounds", "num_boost_round", "n_iter"})
	pm.addMapping("learning_rate", "LearningRate", []string{"shrinkage_rate", "eta"})
	pm.addMapping("num_leaves", "NumLeaves", []string{"num_leaf", "max_leaves", "max_leaf"})
	pm.addMapping("max_depth", "MaxDepth", []string{})
	pm.addMapping("min_child_samples", "MinDataInLeaf", []string{"min_data_in_leaf", "min_data", "min_child_weight"})

	// Tree Growth Parameters
	pm.addMapping("min_split_gain", "MinGainToSplit", []string{"min_gain_to_split"})
	pm.addMapping("max_bin", "MaxBin", []string{})
	pm.addMapping("min_data_in_bin", "MinDataInBin", []string{})

	// Regularization Parameters
	pm.addMapping("reg_alpha", "Alpha", []string{"lambda_l1", "l1_regularization"})
	pm.addMapping("reg_lambda", "Lambda", []string{"lambda_l2", "l2_regularization", "lambda"})
	pm.addMapping("min_child_weight", "MinSumHessianInLeaf", []string{"min_sum_hessian_in_leaf", "min_sum_hessian", "min_hessian"})

	// Sampling Parameters
	pm.addMapping("subsample", "BaggingFraction", []string{"bagging_fraction", "sub_row", "subsample_freq"})
	pm.addMapping("subsample_freq", "BaggingFreq", []string{"bagging_freq"})
	pm.addMapping("colsample_bytree", "FeatureFraction", []string{"feature_fraction", "sub_feature", "colsample_bynode"})
	pm.addMapping("colsample_bynode", "FeatureFractionByNode", []string{"feature_fraction_bynode"})

	// Objective and Task Parameters
	pm.addMapping("objective", "Objective", []string{"objective_type", "app", "application", "loss"})
	pm.addMapping("num_class", "NumClass", []string{"num_classes"})
	pm.addMapping("metric", "Metric", []string{"metrics", "metric_types"})
	pm.addMapping("boosting_type", "Boosting", []string{"boosting", "boost"})

	// Training Control Parameters
	pm.addMapping("random_state", "Seed", []string{"seed", "random_seed", "data_random_seed"})
	pm.addMapping("n_jobs", "NumThreads", []string{"num_threads", "num_thread", "nthread"})
	pm.addMapping("importance_type", "ImportanceType", []string{})
	pm.addMapping("deterministic", "Deterministic", []string{"force_row_wise", "force_col_wise"})
	pm.addMapping("verbosity", "Verbosity", []string{"verbose"})
	pm.addMapping("early_stopping_rounds", "EarlyStopping", []string{"early_stopping_round", "early_stopping", "n_iter_no_change"})

	// Categorical Features
	pm.addMapping("categorical_feature", "CategoricalFeatures", []string{"cat_feature", "categorical_column", "cat_column"})
	pm.addMapping("cat_smooth", "CatSmooth", []string{"categorical_smooth"})
	pm.addMapping("cat_l2", "CatL2", []string{})
	pm.addMapping("max_cat_to_onehot", "MaxCatToOnehot", []string{})

	// Advanced Parameters
	pm.addMapping("device_type", "DeviceType", []string{"device"})
	pm.addMapping("tree_learner", "TreeLearner", []string{"tree_type", "tree"})
	pm.addMapping("num_parallel_tree", "NumParallelTree", []string{})
	pm.addMapping("monotone_constraints", "MonotoneConstraints", []string{"monotone_constraint"})
	pm.addMapping("interaction_constraints", "InteractionConstraints", []string{})

	// DART specific parameters
	pm.addMapping("drop_rate", "DropRate", []string{"rate_drop"})
	pm.addMapping("max_drop", "MaxDrop", []string{"max_dropout"})
	pm.addMapping("skip_drop", "SkipDrop", []string{"skip_dropout"})
	pm.addMapping("uniform_drop", "UniformDrop", []string{})
	pm.addMapping("xgboost_dart_mode", "XGBoostDartMode", []string{})
	pm.addMapping("drop_seed", "DropSeed", []string{})

	// GOSS specific parameters
	pm.addMapping("top_rate", "TopRate", []string{})
	pm.addMapping("other_rate", "OtherRate", []string{})

	// Feature parameters
	pm.addMapping("max_delta_step", "MaxDeltaStep", []string{})
	pm.addMapping("scale_pos_weight", "ScalePosWeight", []string{})
	pm.addMapping("min_gain_to_split", "MinGainToSplit", []string{})

	// Linear tree parameters
	pm.addMapping("linear_tree", "LinearTree", []string{})
	pm.addMapping("linear_lambda", "LinearLambda", []string{})

	// Network parameters (for distributed training)
	pm.addMapping("num_machines", "NumMachines", []string{"num_machine"})
	pm.addMapping("local_listen_port", "LocalListenPort", []string{"local_port"})
	pm.addMapping("time_out", "TimeOut", []string{"network_timeout"})
	pm.addMapping("machine_list_file", "MachineListFile", []string{"machine_list_filename", "machine_list", "mlist"})
}

// addMapping adds a parameter mapping with its aliases
func (pm *ParameterMapper) addMapping(pythonName, goName string, aliases []string) {
	// Add primary mapping
	pm.pythonToGo[pythonName] = goName
	pm.goToPython[goName] = pythonName

	// Add all aliases
	for _, alias := range aliases {
		pm.aliases[alias] = pythonName
		pm.pythonToGo[alias] = goName
	}
}

// initializeDefaults sets default values for all parameters
func (pm *ParameterMapper) initializeDefaults() {
	// Core defaults
	pm.defaults["n_estimators"] = 100
	pm.defaults["learning_rate"] = 0.1
	pm.defaults["num_leaves"] = 31
	pm.defaults["max_depth"] = -1
	pm.defaults["min_child_samples"] = 20

	// Tree growth defaults
	pm.defaults["min_split_gain"] = 0.0
	pm.defaults["max_bin"] = 255
	pm.defaults["min_data_in_bin"] = 3

	// Regularization defaults
	pm.defaults["reg_alpha"] = 0.0
	pm.defaults["reg_lambda"] = 0.0
	pm.defaults["min_child_weight"] = 1e-3

	// Sampling defaults
	pm.defaults["subsample"] = 1.0
	pm.defaults["subsample_freq"] = 0
	pm.defaults["colsample_bytree"] = 1.0
	pm.defaults["colsample_bynode"] = 1.0

	// Objective defaults
	pm.defaults["objective"] = "regression"
	pm.defaults["num_class"] = 1
	pm.defaults["boosting_type"] = "gbdt"
	pm.defaults["metric"] = ""

	// Training control defaults
	pm.defaults["random_state"] = 0
	pm.defaults["n_jobs"] = -1
	pm.defaults["importance_type"] = "gain"
	pm.defaults["deterministic"] = false
	pm.defaults["verbosity"] = -1
	pm.defaults["early_stopping_rounds"] = 0

	// Advanced defaults
	pm.defaults["device_type"] = "cpu"
	pm.defaults["tree_learner"] = "serial"
	pm.defaults["num_parallel_tree"] = 1

	// DART defaults
	pm.defaults["drop_rate"] = 0.1
	pm.defaults["max_drop"] = 50
	pm.defaults["skip_drop"] = 0.5
	pm.defaults["uniform_drop"] = false
	pm.defaults["xgboost_dart_mode"] = false
	pm.defaults["drop_seed"] = 4

	// GOSS defaults
	pm.defaults["top_rate"] = 0.2
	pm.defaults["other_rate"] = 0.1

	// Other defaults
	pm.defaults["max_delta_step"] = 0.0
	pm.defaults["scale_pos_weight"] = 1.0
	pm.defaults["linear_tree"] = false
	pm.defaults["linear_lambda"] = 0.0
}

// MapPythonToGo converts Python parameter names to Go field names
func (pm *ParameterMapper) MapPythonToGo(pythonParams map[string]interface{}) map[string]interface{} {
	goParams := make(map[string]interface{})

	for key, value := range pythonParams {
		// Check for direct mapping
		if goName, ok := pm.pythonToGo[key]; ok {
			goParams[goName] = value
			continue
		}

		// Check for alias
		if canonicalName, ok := pm.aliases[key]; ok {
			if goName, ok := pm.pythonToGo[canonicalName]; ok {
				goParams[goName] = value
			}
			continue
		}

		// If no mapping found, use the original key (for forward compatibility)
		goParams[key] = value
	}

	return goParams
}

// MapGoToPython converts Go field names to Python parameter names
func (pm *ParameterMapper) MapGoToPython(goParams map[string]interface{}) map[string]interface{} {
	pythonParams := make(map[string]interface{})

	for key, value := range goParams {
		// Check for direct mapping
		if pythonName, ok := pm.goToPython[key]; ok {
			pythonParams[pythonName] = value
			continue
		}

		// If no mapping found, convert to snake_case
		pythonParams[toSnakeCase(key)] = value
	}

	return pythonParams
}

// GetDefault returns the default value for a parameter
func (pm *ParameterMapper) GetDefault(paramName string) (interface{}, bool) {
	// Check direct name
	if val, ok := pm.defaults[paramName]; ok {
		return val, true
	}

	// Check if it's an alias
	if canonicalName, ok := pm.aliases[paramName]; ok {
		if val, ok := pm.defaults[canonicalName]; ok {
			return val, true
		}
	}

	return nil, false
}

// ValidateObjective validates and normalizes objective function names
func (pm *ParameterMapper) ValidateObjective(objective string) (string, error) {
	// Normalize to lowercase
	objective = strings.ToLower(objective)

	// Map common aliases to canonical names
	objectiveMap := map[string]string{
		"regression":          "regression",
		"regression_l2":       "regression",
		"l2":                  "regression",
		"mean_squared_error":  "regression",
		"mse":                 "regression",
		"regression_l1":       "regression_l1",
		"l1":                  "regression_l1",
		"mean_absolute_error": "regression_l1",
		"mae":                 "regression_l1",
		"huber":               "huber",
		"fair":                "fair",
		"poisson":             "poisson",
		"quantile":            "quantile",
		"mape":                "mape",
		"gamma":               "gamma",
		"tweedie":             "tweedie",

		// Classification objectives
		"binary":              "binary",
		"binary_logloss":      "binary",
		"binary_crossentropy": "binary",
		"multiclass":          "multiclass",
		"softmax":             "multiclass",
		"multiclassova":       "multiclassova",
		"multiclass_ova":      "multiclassova",
		"ova":                 "multiclassova",
		"ovr":                 "multiclassova",

		// Ranking objectives
		"lambdarank":   "lambdarank",
		"rank_xendcg":  "rank_xendcg",
		"xendcg":       "rank_xendcg",
		"xe_ndcg":      "rank_xendcg",
		"xe_ndcg_mart": "rank_xendcg",
		"xendcg_mart":  "rank_xendcg",
	}

	if canonical, ok := objectiveMap[objective]; ok {
		return canonical, nil
	}

	return "", fmt.Errorf("unknown objective: %s", objective)
}

// ValidateBoostingType validates boosting type
func (pm *ParameterMapper) ValidateBoostingType(boostingType string) (string, error) {
	boostingType = strings.ToLower(boostingType)

	validTypes := map[string]bool{
		"gbdt": true,
		"dart": true,
		"goss": true,
		"rf":   true, // Random Forest
	}

	if validTypes[boostingType] {
		return boostingType, nil
	}

	return "", fmt.Errorf("unknown boosting type: %s", boostingType)
}

// ApplyDefaults applies default values to missing parameters
func (pm *ParameterMapper) ApplyDefaults(params map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{})

	// Copy existing parameters
	for k, v := range params {
		result[k] = v
	}

	// Apply defaults for missing parameters
	for key, defaultValue := range pm.defaults {
		if _, exists := result[key]; !exists {
			result[key] = defaultValue
		}
	}

	return result
}

// Helper function to convert CamelCase to snake_case
func toSnakeCase(s string) string {
	var result strings.Builder
	for i, r := range s {
		if i > 0 && r >= 'A' && r <= 'Z' {
			result.WriteRune('_')
		}
		result.WriteRune(r)
	}
	return strings.ToLower(result.String())
}
