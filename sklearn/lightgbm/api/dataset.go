package api

import (
	"fmt"

	scigoErrors "github.com/YuminosukeSato/scigo/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

// Dataset represents a LightGBM dataset, similar to Python's lgb.Dataset
// This structure holds the data, labels, and various parameters needed for training
type Dataset struct {
	// Core data
	Data  mat.Matrix    // Feature matrix (n_samples × n_features)
	Label mat.Matrix    // Target values (n_samples × 1 for regression/binary, n_samples × n_classes for multiclass)
	Weight *mat.VecDense // Sample weights (optional)
	
	// Metadata
	FeatureNames []string               // Names of features
	CategoricalFeatures []int            // Indices of categorical features
	Reference    *Dataset               // Reference dataset for consistent binning
	Params       map[string]interface{} // Dataset-specific parameters
	
	// Internal state
	nSamples  int
	nFeatures int
	isBinary  bool
	nClasses  int
	
	// Binning information (for histogram-based algorithms)
	binMappers []BinMapper // Bin mapping for each feature
	
	// Validation flag
	isValid bool
}

// BinMapper handles the discretization of continuous features into bins
type BinMapper struct {
	FeatureIndex int       // Index of the feature
	BinBoundaries []float64 // Boundaries for bins
	NumBins      int       // Number of bins
	IsCategorical bool     // Whether this feature is categorical
}

// NewDataset creates a new Dataset instance similar to lgb.Dataset(data, label=y)
func NewDataset(data mat.Matrix, label mat.Matrix, options ...DatasetOption) (*Dataset, error) {
	if data == nil {
		return nil, scigoErrors.NewValueError("NewDataset", "data cannot be nil")
	}
	
	rows, cols := data.Dims()
	if rows == 0 || cols == 0 {
		return nil, scigoErrors.NewValueError("NewDataset", "data cannot be empty")
	}
	
	ds := &Dataset{
		Data:      data,
		Label:     label,
		nSamples:  rows,
		nFeatures: cols,
		Params:    make(map[string]interface{}),
		isValid:   true,
	}
	
	// Process labels if provided
	if label != nil {
		if err := ds.validateLabel(); err != nil {
			return nil, err
		}
	}
	
	// Apply options
	for _, opt := range options {
		opt(ds)
	}
	
	// Initialize feature names if not provided
	if len(ds.FeatureNames) == 0 {
		ds.FeatureNames = make([]string, cols)
		for i := 0; i < cols; i++ {
			ds.FeatureNames[i] = fmt.Sprintf("Column_%d", i)
		}
	}
	
	return ds, nil
}

// DatasetOption is a functional option for configuring Dataset
type DatasetOption func(*Dataset)

// WithWeight sets sample weights
func WithWeight(weight *mat.VecDense) DatasetOption {
	return func(ds *Dataset) {
		ds.Weight = weight
	}
}

// WithFeatureNames sets feature names
func WithFeatureNames(names []string) DatasetOption {
	return func(ds *Dataset) {
		ds.FeatureNames = names
	}
}

// WithCategoricalFeatures sets categorical feature indices
func WithCategoricalFeatures(indices []int) DatasetOption {
	return func(ds *Dataset) {
		ds.CategoricalFeatures = indices
	}
}

// WithReference sets a reference dataset for consistent binning
func WithReference(ref *Dataset) DatasetOption {
	return func(ds *Dataset) {
		ds.Reference = ref
		if ref != nil && ref.binMappers != nil {
			// Copy bin mappers from reference
			ds.binMappers = make([]BinMapper, len(ref.binMappers))
			copy(ds.binMappers, ref.binMappers)
		}
	}
}

// WithParams sets dataset-specific parameters
func WithParams(params map[string]interface{}) DatasetOption {
	return func(ds *Dataset) {
		for k, v := range params {
			ds.Params[k] = v
		}
	}
}

// validateLabel validates the label matrix and determines the task type
func (ds *Dataset) validateLabel() error {
	if ds.Label == nil {
		return nil
	}
	
	labelRows, labelCols := ds.Label.Dims()
	
	if labelRows != ds.nSamples {
		return scigoErrors.NewDimensionError(
			"Dataset.validateLabel",
			ds.nSamples,
			labelRows,
			0,
		)
	}
	
	// Determine task type based on label shape and values
	if labelCols == 1 {
		// Could be regression or binary classification
		ds.detectBinaryClassification()
	} else {
		// Multiclass with one-hot encoding
		ds.nClasses = labelCols
	}
	
	return nil
}

// detectBinaryClassification checks if labels are binary
func (ds *Dataset) detectBinaryClassification() {
	rows, _ := ds.Label.Dims()
	uniqueVals := make(map[float64]bool)
	
	for i := 0; i < rows; i++ {
		val := ds.Label.At(i, 0)
		uniqueVals[val] = true
		
		// If we have many unique values, it's likely regression
		if len(uniqueVals) > rows/4 {
			ds.isBinary = false
			return
		}
	}
	
	// Check if all values are 0 or 1
	ds.isBinary = true
	for val := range uniqueVals {
		if val != 0.0 && val != 1.0 {
			ds.isBinary = false
			break
		}
	}
	
	if ds.isBinary {
		ds.nClasses = 2
	}
}

// GetField returns a specific field from the dataset
func (ds *Dataset) GetField(fieldName string) (interface{}, error) {
	switch fieldName {
	case "label":
		return ds.Label, nil
	case "weight":
		return ds.Weight, nil
	case "data":
		return ds.Data, nil
	case "feature_name":
		return ds.FeatureNames, nil
	case "categorical_feature":
		return ds.CategoricalFeatures, nil
	case "n_samples":
		return ds.nSamples, nil
	case "n_features":
		return ds.nFeatures, nil
	default:
		if val, ok := ds.Params[fieldName]; ok {
			return val, nil
		}
		return nil, fmt.Errorf("field '%s' not found", fieldName)
	}
}

// SetField sets a specific field in the dataset
func (ds *Dataset) SetField(fieldName string, value interface{}) error {
	switch fieldName {
	case "label":
		if label, ok := value.(mat.Matrix); ok {
			ds.Label = label
			return ds.validateLabel()
		}
		return fmt.Errorf("invalid type for label field")
	case "weight":
		if weight, ok := value.(*mat.VecDense); ok {
			ds.Weight = weight
			return nil
		}
		return fmt.Errorf("invalid type for weight field")
	case "feature_name":
		if names, ok := value.([]string); ok {
			ds.FeatureNames = names
			return nil
		}
		return fmt.Errorf("invalid type for feature_name field")
	case "categorical_feature":
		if indices, ok := value.([]int); ok {
			ds.CategoricalFeatures = indices
			return nil
		}
		return fmt.Errorf("invalid type for categorical_feature field")
	default:
		ds.Params[fieldName] = value
		return nil
	}
}

// NumData returns the number of samples in the dataset
func (ds *Dataset) NumData() int {
	return ds.nSamples
}

// NumFeature returns the number of features in the dataset
func (ds *Dataset) NumFeature() int {
	return ds.nFeatures
}

// GetLabel returns the label matrix
func (ds *Dataset) GetLabel() mat.Matrix {
	return ds.Label
}

// GetData returns the data matrix
func (ds *Dataset) GetData() mat.Matrix {
	return ds.Data
}

// GetWeight returns the weight vector
func (ds *Dataset) GetWeight() *mat.VecDense {
	return ds.Weight
}

// GetSubset creates a subset of the dataset with specified indices
func (ds *Dataset) GetSubset(indices []int) (*Dataset, error) {
	if len(indices) == 0 {
		return nil, scigoErrors.NewValueError("GetSubset", "indices cannot be empty")
	}
	
	// Create subset data matrix
	subsetData := mat.NewDense(len(indices), ds.nFeatures, nil)
	for i, idx := range indices {
		if idx < 0 || idx >= ds.nSamples {
			return nil, fmt.Errorf("index %d out of range [0, %d)", idx, ds.nSamples)
		}
		for j := 0; j < ds.nFeatures; j++ {
			subsetData.Set(i, j, ds.Data.At(idx, j))
		}
	}
	
	// Create subset label if exists
	var subsetLabel mat.Matrix
	if ds.Label != nil {
		_, labelCols := ds.Label.Dims()
		subsetLabel = mat.NewDense(len(indices), labelCols, nil)
		for i, idx := range indices {
			for j := 0; j < labelCols; j++ {
				subsetLabel.Set(i, j, ds.Label.At(idx, j))
			}
		}
	}
	
	// Create subset weight if exists
	var subsetWeight *mat.VecDense
	if ds.Weight != nil {
		weightData := make([]float64, len(indices))
		for i, idx := range indices {
			weightData[i] = ds.Weight.AtVec(idx)
		}
		subsetWeight = mat.NewVecDense(len(indices), weightData)
	}
	
	// Create new dataset with subset
	subset := &Dataset{
		Data:                subsetData,
		Label:               subsetLabel,
		Weight:              subsetWeight,
		FeatureNames:        ds.FeatureNames,
		CategoricalFeatures: ds.CategoricalFeatures,
		Reference:           ds.Reference,
		Params:              make(map[string]interface{}),
		nSamples:            len(indices),
		nFeatures:           ds.nFeatures,
		isBinary:            ds.isBinary,
		nClasses:            ds.nClasses,
		binMappers:          ds.binMappers,
		isValid:             true,
	}
	
	// Copy parameters
	for k, v := range ds.Params {
		subset.Params[k] = v
	}
	
	return subset, nil
}

// CreateValidDatasets creates validation datasets from a list of datasets
// This is similar to Python's valid_sets parameter in lgb.train()
func CreateValidDatasets(datasets ...*Dataset) ([]*Dataset, []string, error) {
	if len(datasets) == 0 {
		return nil, nil, nil
	}
	
	validSets := make([]*Dataset, 0, len(datasets))
	validNames := make([]string, 0, len(datasets))
	
	for i, ds := range datasets {
		if ds == nil {
			continue
		}
		if !ds.isValid {
			return nil, nil, fmt.Errorf("dataset %d is not valid", i)
		}
		validSets = append(validSets, ds)
		
		// Generate name if not specified
		name := fmt.Sprintf("valid_%d", i)
		if nameParam, ok := ds.Params["name"].(string); ok {
			name = nameParam
		}
		validNames = append(validNames, name)
	}
	
	return validSets, validNames, nil
}

// ConstructFromFile creates a Dataset from a file (LibSVM, CSV, etc.)
// This is similar to Python's lgb.Dataset(filename)
func ConstructFromFile(filename string, params map[string]interface{}) (*Dataset, error) {
	// This would need implementation based on file format detection
	// For now, return an error indicating it's not yet implemented
	return nil, fmt.Errorf("ConstructFromFile not yet implemented")
}