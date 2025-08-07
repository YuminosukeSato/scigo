package model

import (
	"encoding/gob"
	"fmt"
	"io"
	"os"
)

// SaveModel saves a model to a file
//
// Parameters:
//   - model: The model to save (struct with embedded BaseEstimator)
//   - filename: The file path to save to
//
// Returns:
//   - error: Error if saving fails
//
// Example:
//
//	var reg linear.Regression
//	// ... train the model ...
//	err := model.SaveModel(&reg, "model.gob")
func SaveModel(model interface{}, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer func() { _ = file.Close() }()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(model); err != nil {
		return fmt.Errorf("failed to encode model: %w", err)
	}

	return nil
}

// LoadModel loads a model from a file
//
// Parameters:
//   - model: The target model (pointer to struct with embedded BaseEstimator)
//   - filename: The file path to load from
//
// Returns:
//   - error: Error if loading fails
//
// Example:
//
//	var reg linear.Regression
//	err := model.LoadModel(&reg, "model.gob")
func LoadModel(model interface{}, filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer func() { _ = file.Close() }()

	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(model); err != nil {
		return fmt.Errorf("failed to decode model: %w", err)
	}

	return nil
}

// SaveModelToWriter saves a model to an io.Writer
//
// Parameters:
//   - model: The model to save
//   - w: The target Writer
//
// Returns:
//   - error: Error if saving fails
func SaveModelToWriter(model interface{}, w io.Writer) error {
	encoder := gob.NewEncoder(w)
	if err := encoder.Encode(model); err != nil {
		return fmt.Errorf("failed to encode model: %w", err)
	}
	return nil
}

// LoadModelFromReader loads a model from an io.Reader
//
// Parameters:
//   - model: The target model (pointer)
//   - r: The source Reader
//
// Returns:
//   - error: Error if loading fails
func LoadModelFromReader(model interface{}, r io.Reader) error {
	decoder := gob.NewDecoder(r)
	if err := decoder.Decode(model); err != nil {
		return fmt.Errorf("failed to decode model: %w", err)
	}
	return nil
}
