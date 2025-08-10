// Package lightgbm provides a pure Go implementation compatible with LightGBM C API
package lightgbm

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// C API compatible types and structures

// LGBMDataset represents a dataset for training/prediction
type LGBMDataset struct {
	// Data matrix (samples x features)
	Data *mat.Dense
	// Labels for supervised learning
	Label []float32
	// Feature names
	FeatureNames []string
	// Categorical feature indices
	CategoricalFeatures []int
	// Internal statistics
	numData     int
	numFeatures int
}

// LGBMBooster represents a gradient boosting model
type LGBMBooster struct {
	// Training parameters
	Params map[string]string
	// Training dataset
	TrainSet *LGBMDataset
	// Trees in the model
	Trees []CAPITree
	// Current iteration
	CurrentIter int
	// Feature importance
	FeatureImportance []float64
	// Objective function
	Objective string
	// Number of classes (for multiclass)
	NumClass int
	// Initial score (base prediction)
	InitScore float64
}

// DatasetCreateFromMat creates a new dataset from a matrix
// This is equivalent to LGBM_DatasetCreateFromMat in C API
func DatasetCreateFromMat(data []float32, nrow, ncol int, isRowMajor bool, label []float32) (*LGBMDataset, error) {
	if len(data) != nrow*ncol {
		return nil, fmt.Errorf("data size mismatch: expected %d, got %d", nrow*ncol, len(data))
	}

	if label != nil && len(label) != nrow {
		return nil, fmt.Errorf("label size mismatch: expected %d, got %d", nrow, len(label))
	}

	// Convert to gonum matrix
	matData := make([]float64, len(data))
	for i, v := range data {
		matData[i] = float64(v)
	}

	var dense *mat.Dense
	if isRowMajor {
		dense = mat.NewDense(nrow, ncol, matData)
	} else {
		// Column major - need to transpose
		temp := mat.NewDense(ncol, nrow, matData)
		dense = mat.DenseCopyOf(temp.T())
	}

	dataset := &LGBMDataset{
		Data:        dense,
		Label:       label,
		numData:     nrow,
		numFeatures: ncol,
	}

	return dataset, nil
}

// DatasetFree frees the dataset (for API compatibility)
func DatasetFree(dataset *LGBMDataset) error {
	// In Go, garbage collection handles this
	// This function exists for C API compatibility
	return nil
}

// BoosterCreate creates a new booster
// This is equivalent to LGBM_BoosterCreate in C API
func BoosterCreate(trainData *LGBMDataset, parameters string) (*LGBMBooster, error) {
	if trainData == nil {
		return nil, fmt.Errorf("train dataset is required")
	}

	// Parse parameters
	params := parseParameters(parameters)

	// Set defaults if not provided
	setDefaultParams(params)

	// Determine objective and num_class
	objective := params["objective"]
	numClass := 1
	if objective == "multiclass" || objective == "multiclassova" {
		if nc, ok := params["num_class"]; ok {
			numClass, _ = strconv.Atoi(nc)
		}
	}

	// Calculate initial score
	initScore := calculateInitScore(trainData.Label, objective)

	booster := &LGBMBooster{
		Params:      params,
		TrainSet:    trainData,
		Trees:       make([]CAPITree, 0),
		CurrentIter: 0,
		Objective:   objective,
		NumClass:    numClass,
		InitScore:   initScore,
	}

	return booster, nil
}

// BoosterFree frees the booster (for API compatibility)
func BoosterFree(booster *LGBMBooster) error {
	// In Go, garbage collection handles this
	// This function exists for C API compatibility
	return nil
}

// BoosterUpdateOneIter updates the model for one iteration
// This is equivalent to LGBM_BoosterUpdateOneIter in C API
func BoosterUpdateOneIter(booster *LGBMBooster) error {
	if booster == nil || booster.TrainSet == nil {
		return fmt.Errorf("invalid booster or training set")
	}

	// Get current predictions
	predictions := booster.predictInternal(booster.TrainSet.Data)

	// Calculate gradients and hessians based on objective
	gradients, hessians := calculateGradients(
		predictions,
		booster.TrainSet.Label,
		booster.Objective,
	)

	// Build a new tree
	tree := buildTree(
		booster.TrainSet.Data,
		gradients,
		hessians,
		booster.Params,
	)

	// Add tree to the model
	booster.Trees = append(booster.Trees, tree)
	booster.CurrentIter++

	return nil
}

// BoosterPredictForMat makes predictions for a data matrix
// This is equivalent to LGBM_BoosterPredictForMat in C API
func BoosterPredictForMat(booster *LGBMBooster, data []float32, nrow, ncol int, isRowMajor bool) ([]float64, error) {
	if booster == nil {
		return nil, fmt.Errorf("booster is nil")
	}

	// Convert input data to matrix
	matData := make([]float64, len(data))
	for i, v := range data {
		matData[i] = float64(v)
	}

	var dense *mat.Dense
	if isRowMajor {
		dense = mat.NewDense(nrow, ncol, matData)
	} else {
		temp := mat.NewDense(ncol, nrow, matData)
		dense = mat.DenseCopyOf(temp.T())
	}

	// Make predictions
	predictions := booster.predictInternal(dense)
	return predictions, nil
}

// Internal helper functions

func parseParameters(parameters string) map[string]string {
	params := make(map[string]string)
	if parameters == "" {
		return params
	}

	pairs := strings.Split(parameters, " ")
	for _, pair := range pairs {
		kv := strings.SplitN(pair, "=", 2)
		if len(kv) == 2 {
			params[kv[0]] = kv[1]
		}
	}

	return params
}

func setDefaultParams(params map[string]string) {
	defaults := map[string]string{
		"objective":         "regression",
		"num_leaves":        "31",
		"max_depth":         "-1",
		"learning_rate":     "0.1",
		"num_iterations":    "100",
		"feature_fraction":  "1.0",
		"bagging_fraction":  "1.0",
		"lambda_l2":         "0.0",
		"lambda_l1":         "0.0",
		"min_data_in_leaf":  "20",
		"min_gain_to_split": "0.0",
		"num_threads":       "1",
	}

	for k, v := range defaults {
		if _, exists := params[k]; !exists {
			params[k] = v
		}
	}
}

func calculateInitScore(labels []float32, objective string) float64 {
	if len(labels) == 0 {
		return 0.0
	}

	switch objective {
	case "regression", "regression_l2", "mean_squared_error", "mse":
		// For regression, use mean of labels
		sum := float64(0)
		for _, v := range labels {
			sum += float64(v)
		}
		return sum / float64(len(labels))

	case "binary", "binary_logloss":
		// For binary classification, use log odds
		positive := 0
		for _, v := range labels {
			if v > 0.5 {
				positive++
			}
		}
		ratio := float64(positive) / float64(len(labels))
		if ratio <= 0 {
			ratio = 1e-10
		} else if ratio >= 1 {
			ratio = 1 - 1e-10
		}
		return logit(ratio)

	default:
		return 0.0
	}
}

func logit(p float64) float64 {
	return math.Log(p / (1 - p))
}

func (b *LGBMBooster) predictInternal(data *mat.Dense) []float64 {
	nrow, _ := data.Dims()
	predictions := make([]float64, nrow)

	// Start with initial score
	for i := range predictions {
		predictions[i] = b.InitScore
	}

	// Add contribution from each tree
	learningRate, _ := strconv.ParseFloat(b.Params["learning_rate"], 64)

	for _, tree := range b.Trees {
		for i := 0; i < nrow; i++ {
			row := mat.Row(nil, i, data)
			leafValue := tree.predict(row)
			predictions[i] += leafValue * learningRate
		}
	}

	return predictions
}

func calculateGradients(predictions []float64, labels []float32, objective string) ([]float64, []float64) {
	n := len(predictions)
	gradients := make([]float64, n)
	hessians := make([]float64, n)

	switch objective {
	case "regression", "regression_l2", "mean_squared_error", "mse":
		// Squared loss: L = (y - p)^2
		// Gradient: -2(y - p)
		// Hessian: 2
		for i := range predictions {
			residual := float64(labels[i]) - predictions[i]
			gradients[i] = 2.0 * residual // Note: LightGBM uses negative gradient
			hessians[i] = 2.0
		}

	case "binary", "binary_logloss":
		// Binary log loss
		for i := range predictions {
			p := capiSigmoid(predictions[i])
			gradients[i] = p - float64(labels[i])
			hessians[i] = p * (1.0 - p)
		}

	default:
		// Default to squared loss
		for i := range predictions {
			residual := float64(labels[i]) - predictions[i]
			gradients[i] = 2.0 * residual
			hessians[i] = 2.0
		}
	}

	return gradients, hessians
}

func capiSigmoid(x float64) float64 {
	if x > 0 {
		expNegX := math.Exp(-x)
		return 1.0 / (1.0 + expNegX)
	} else {
		expX := math.Exp(x)
		return expX / (1.0 + expX)
	}
}

