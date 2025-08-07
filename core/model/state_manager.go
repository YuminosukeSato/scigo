// Package model provides state management for machine learning models.
package model

import (
	"fmt"
	"sync"
)

// StateManager manages the fitted state of a model in a thread-safe manner.
// It replaces the BaseEstimator embedding pattern with composition.
type StateManager struct {
	Fitted bool // Public for gob encoding
	mu     sync.RWMutex

	// Optional metadata - Public for gob encoding
	NFeatures int
	NSamples  int
}

// NewStateManager creates a new StateManager instance.
func NewStateManager() *StateManager {
	return &StateManager{
		Fitted: false,
	}
}

// IsFitted returns whether the model has been fitted.
func (s *StateManager) IsFitted() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.Fitted
}

// SetFitted marks the model as fitted.
func (s *StateManager) SetFitted() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Fitted = true
}

// Reset resets the fitted state.
func (s *StateManager) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Fitted = false
	s.NFeatures = 0
	s.NSamples = 0
}

// SetDimensions sets the number of features and samples seen during fitting.
func (s *StateManager) SetDimensions(nFeatures, nSamples int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.NFeatures = nFeatures
	s.NSamples = nSamples
}

// GetDimensions returns the number of features and samples seen during fitting.
func (s *StateManager) GetDimensions() (nFeatures, nSamples int) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.NFeatures, s.NSamples
}

// RequireFitted returns an error if the model has not been fitted.
func (s *StateManager) RequireFitted() error {
	if !s.IsFitted() {
		return fmt.Errorf("model has not been fitted yet. Call Fit() first")
	}
	return nil
}

// ModelState represents the complete state of a model.
// This can be used for serialization and debugging.
type ModelState struct {
	Fitted    bool                   `json:"fitted"`
	NFeatures int                    `json:"n_features,omitempty"`
	NSamples  int                    `json:"n_samples,omitempty"`
	Params    map[string]interface{} `json:"params,omitempty"`
}

// GetState returns the current state as a ModelState struct.
func (s *StateManager) GetState() ModelState {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return ModelState{
		Fitted:    s.Fitted,
		NFeatures: s.NFeatures,
		NSamples:  s.NSamples,
	}
}

// SetState sets the state from a ModelState struct.
func (s *StateManager) SetState(state ModelState) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.Fitted = state.Fitted
	s.NFeatures = state.NFeatures
	s.NSamples = state.NSamples
}

// WithState is a helper function that executes a function with the state locked for reading.
func (s *StateManager) WithState(fn func() error) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return fn()
}

// WithStateMut is a helper function that executes a function with the state locked for writing.
func (s *StateManager) WithStateMut(fn func() error) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return fn()
}
