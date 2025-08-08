// +build capi

package lightgbm_capi

/*
#cgo LDFLAGS: -llightgbm -lstdc++
#cgo CFLAGS: -I/usr/local/include

#include <stdlib.h>
#include <string.h>
#include <LightGBM/c_api.h>

// Helper function to create C string array
char** makeCharArray(int size) {
    return (char**)malloc(sizeof(char*) * size);
}

// Helper function to set string in array
void setArrayString(char** array, int index, char* str) {
    array[index] = str;
}

// Helper function to free string array
void freeCharArray(char** array, int size) {
    free(array);
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// BoosterHandle wraps the C API booster handle
type BoosterHandle struct {
	handle C.BoosterHandle
}

// DatasetHandle wraps the C API dataset handle
type DatasetHandle struct {
	handle C.DatasetHandle
}

// LastError returns the last error message from LightGBM C API
func LastError() error {
	errMsg := C.LGBM_GetLastError()
	if errMsg != nil {
		return fmt.Errorf("LightGBM error: %s", C.GoString(errMsg))
	}
	return nil
}

// CreateDatasetFromMat creates a dataset from a matrix
func CreateDatasetFromMat(data []float64, nrow, ncol int, params string) (*DatasetHandle, error) {
	var handle C.DatasetHandle
	
	cParams := C.CString(params)
	defer C.free(unsafe.Pointer(cParams))
	
	ret := C.LGBM_DatasetCreateFromMat(
		unsafe.Pointer(&data[0]),
		C.C_API_DTYPE_FLOAT64,
		C.int32_t(nrow),
		C.int32_t(ncol),
		C.int(1), // is_row_major
		cParams,
		nil,
		&handle,
	)
	
	if ret != 0 {
		return nil, LastError()
	}
	
	return &DatasetHandle{handle: handle}, nil
}

// SetField sets a field in the dataset (e.g., "label")
func (d *DatasetHandle) SetField(fieldName string, data []float32) error {
	cFieldName := C.CString(fieldName)
	defer C.free(unsafe.Pointer(cFieldName))
	
	ret := C.LGBM_DatasetSetField(
		d.handle,
		cFieldName,
		unsafe.Pointer(&data[0]),
		C.int(len(data)),
		C.C_API_DTYPE_FLOAT32,
	)
	
	if ret != 0 {
		return LastError()
	}
	
	return nil
}

// Free frees the dataset
func (d *DatasetHandle) Free() error {
	ret := C.LGBM_DatasetFree(d.handle)
	if ret != 0 {
		return LastError()
	}
	return nil
}

// BoosterCreate creates a new booster
func BoosterCreate(trainData *DatasetHandle, params string) (*BoosterHandle, error) {
	var handle C.BoosterHandle
	
	cParams := C.CString(params)
	defer C.free(unsafe.Pointer(cParams))
	
	ret := C.LGBM_BoosterCreate(
		trainData.handle,
		cParams,
		&handle,
	)
	
	if ret != 0 {
		return nil, LastError()
	}
	
	return &BoosterHandle{handle: handle}, nil
}

// BoosterCreateFromModelfile loads a booster from file
func BoosterCreateFromModelfile(filename string) (*BoosterHandle, error) {
	var handle C.BoosterHandle
	var outNumIterations C.int
	
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))
	
	ret := C.LGBM_BoosterCreateFromModelfile(
		cFilename,
		&outNumIterations,
		&handle,
	)
	
	if ret != 0 {
		return nil, LastError()
	}
	
	return &BoosterHandle{handle: handle}, nil
}

// UpdateOneIter performs one boosting iteration
func (b *BoosterHandle) UpdateOneIter() error {
	var isFinished C.int
	
	ret := C.LGBM_BoosterUpdateOneIter(b.handle, &isFinished)
	if ret != 0 {
		return LastError()
	}
	
	return nil
}

// PredictForMat makes predictions for a matrix
func (b *BoosterHandle) PredictForMat(data []float64, nrow, ncol int, predictType int, numIteration int) ([]float64, error) {
	var outLen C.int64_t
	
	// Estimate output length
	// For normal prediction: nrow
	// For leaf index: nrow * num_trees
	// For multiclass: nrow * num_class
	estimatedLen := nrow * 10 // Conservative estimate
	outResult := make([]float64, estimatedLen)
	
	cParams := C.CString("")
	defer C.free(unsafe.Pointer(cParams))
	
	ret := C.LGBM_BoosterPredictForMat(
		b.handle,
		unsafe.Pointer(&data[0]),
		C.C_API_DTYPE_FLOAT64,
		C.int32_t(nrow),
		C.int32_t(ncol),
		C.int(1), // is_row_major
		C.int(predictType),
		C.int(numIteration),
		cParams,
		&outLen,
		&outResult[0],
	)
	
	if ret != 0 {
		return nil, LastError()
	}
	
	// Resize to actual output length
	return outResult[:outLen], nil
}

// SaveModel saves the model to a file
func (b *BoosterHandle) SaveModel(filename string, numIteration int, featureImportanceType int) error {
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))
	
	ret := C.LGBM_BoosterSaveModel(
		b.handle,
		C.int(0),     // start_iteration
		C.int(numIteration),
		C.int(featureImportanceType),
		cFilename,
	)
	
	if ret != 0 {
		return LastError()
	}
	
	return nil
}

// DumpModel dumps model to JSON string
func (b *BoosterHandle) DumpModel(numIteration int, featureImportanceType int) (string, error) {
	var outLen C.int64_t
	var outStr *C.char
	
	ret := C.LGBM_BoosterDumpModel(
		b.handle,
		C.int(0), // start_iteration
		C.int(numIteration),
		C.int(featureImportanceType),
		C.int64_t(10*1024*1024), // buffer_len: 10MB
		&outLen,
		unsafe.Pointer(&outStr),
	)
	
	if ret != 0 {
		return "", LastError()
	}
	
	result := C.GoStringN(outStr, C.int(outLen))
	return result, nil
}

// GetNumClasses returns the number of classes for multiclass classification
func (b *BoosterHandle) GetNumClasses() (int, error) {
	var numClass C.int
	
	ret := C.LGBM_BoosterGetNumClasses(b.handle, &numClass)
	if ret != 0 {
		return 0, LastError()
	}
	
	return int(numClass), nil
}

// GetNumFeature returns the number of features
func (b *BoosterHandle) GetNumFeature() (int, error) {
	var numFeature C.int
	
	ret := C.LGBM_BoosterGetNumFeature(b.handle, &numFeature)
	if ret != 0 {
		return 0, LastError()
	}
	
	return int(numFeature), nil
}

// Free frees the booster
func (b *BoosterHandle) Free() error {
	ret := C.LGBM_BoosterFree(b.handle)
	if ret != 0 {
		return LastError()
	}
	return nil
}

// Prediction types
const (
	PredictNormal     = 0
	PredictRawScore   = 1
	PredictLeafIndex  = 2
	PredictContrib    = 3
)