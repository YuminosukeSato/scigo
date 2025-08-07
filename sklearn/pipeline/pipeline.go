// Package pipeline implements scikit-learn compatible Pipeline for chaining transformers and estimators.
// This provides the same API as sklearn.pipeline.Pipeline.
package pipeline

import (
	"fmt"

	"github.com/YuminosukeSato/scigo/core/model"
	"github.com/YuminosukeSato/scigo/pkg/errors"
	"gonum.org/v1/gonum/mat"
	"github.com/YuminosukeSato/scigo/pkg/log"
)

var globalProvider log.LoggerProvider

// Step represents a single step in the pipeline.
// Each step is a tuple of (name, transformer/estimator).
type Step struct {
	Name      string      // Name of this step (for identification)
	Estimator interface{} // Can be Transformer or Estimator
}

// Pipeline chains multiple transforms and optionally a final estimator.
// Intermediate steps must be transformers (i.e., have a transform method).
// The final step can be a transformer or an estimator.
//
// Scikit-learn compatible implementation.
type Pipeline struct {
	// State management using composition
	state  *model.StateManager
	logger log.Logger

	// Pipeline configuration
	steps   []Step // List of (name, transform/estimator) tuples
	memory  string // TODO: Implement caching
	verbose bool   // If True, time elapsed while fitting each step

	// Fitted state
	namedSteps_ map[string]interface{} // Access steps by name
}

// New creates a new Pipeline with the given steps.
// This is equivalent to sklearn.pipeline.Pipeline(steps)
func New(steps ...Step) *Pipeline {
	namedSteps := make(map[string]interface{})
	for _, step := range steps {
		namedSteps[step.Name] = step.Estimator
	}

	pipeline := &Pipeline{
		steps:       steps,
		namedSteps_: namedSteps,
		verbose:     false,
	}

	// Initialize state manager and logger
	pipeline.state = model.NewStateManager()
	if globalProvider == nil {
		globalProvider = log.NewZerologProvider(log.ToLogLevel("info"))
	}
	pipeline.logger = globalProvider.GetLoggerWithName("Pipeline")

	return pipeline
}

// NewPipeline is an alias for New to match sklearn naming conventions
func NewPipeline(steps ...Step) *Pipeline {
	return New(steps...)
}

// Make is a convenience function similar to sklearn.pipeline.make_pipeline
// It automatically generates names for the steps.
func Make(estimators ...interface{}) *Pipeline {
	steps := make([]Step, len(estimators))
	for i, estimator := range estimators {
		// Generate name from type (simplified version)
		name := fmt.Sprintf("step%d", i+1)
		steps[i] = Step{Name: name, Estimator: estimator}
	}
	return New(steps...)
}

// Fit trains the pipeline.
// Fit all the transformers one after the other and transform the
// data, then fit the final estimator.
func (p *Pipeline) Fit(X, y mat.Matrix) error {
	Xt := X
	var err error

	// Fit and transform all steps except the last
	for i := 0; i < len(p.steps)-1; i++ {
		step := p.steps[i]

		// Check if step implements Transformer interface
		transformer, ok := step.Estimator.(model.Transformer)
		if !ok {
			return errors.NewValidationError(
				"pipeline step",
				"all intermediate steps must be transformers",
				step.Name,
			)
		}

		// Fit the transformer (Transformer.Fit only takes X)
		if err = transformer.Fit(Xt); err != nil {
			return errors.Wrap(err, fmt.Sprintf("failed to fit step '%s'", step.Name))
		}

		Xt, err = transformer.Transform(Xt)
		if err != nil {
			return errors.Wrap(err, fmt.Sprintf("failed to transform at step '%s'", step.Name))
		}
	}

	// Fit the final estimator
	if len(p.steps) > 0 {
		finalStep := p.steps[len(p.steps)-1]

		// The final step can be either transformer or estimator
		if fitter, ok := finalStep.Estimator.(interface {
			Fit(mat.Matrix, mat.Matrix) error
		}); ok {
			if err = fitter.Fit(Xt, y); err != nil {
				return errors.Wrap(err, fmt.Sprintf("failed to fit final step '%s'", finalStep.Name))
			}
		} else {
			return errors.NewValidationError(
				"pipeline final step",
				"final step must have Fit method",
				finalStep.Name,
			)
		}
	}

	p.state.SetFitted()
	return nil
}

// Predict applies transforms to the data, and predict with the final estimator.
func (p *Pipeline) Predict(X mat.Matrix) (mat.Matrix, error) {
	if !p.state.IsFitted() {
		return nil, errors.NewNotFittedError("Pipeline", "Predict")
	}

	Xt, err := p.transform(X)
	if err != nil {
		return nil, err
	}

	// Predict with the final estimator
	if len(p.steps) > 0 {
		finalStep := p.steps[len(p.steps)-1]

		if predictor, ok := finalStep.Estimator.(interface {
			Predict(mat.Matrix) (mat.Matrix, error)
		}); ok {
			return predictor.Predict(Xt)
		}

		return nil, errors.NewValidationError(
			"pipeline final step",
			"final step must have Predict method for prediction",
			finalStep.Name,
		)
	}

	return Xt, nil
}

// Transform applies transforms to the data.
// Only valid if the final step is a transformer.
func (p *Pipeline) Transform(X mat.Matrix) (mat.Matrix, error) {
	if !p.state.IsFitted() {
		return nil, errors.NewNotFittedError("Pipeline", "Transform")
	}

	// Transform through all steps including the last
	Xt := X
	var err error

	for _, step := range p.steps {
		transformer, ok := step.Estimator.(model.Transformer)
		if !ok {
			return nil, errors.NewValidationError(
				"pipeline step",
				"all steps must be transformers for Transform",
				step.Name,
			)
		}

		Xt, err = transformer.Transform(Xt)
		if err != nil {
			return nil, errors.Wrap(err, fmt.Sprintf("failed to transform at step '%s'", step.Name))
		}
	}

	return Xt, nil
}

// FitPredict is a convenience method that fits the pipeline and predicts.
// Equivalent to calling Fit followed by Predict.
func (p *Pipeline) FitPredict(X, y mat.Matrix) (mat.Matrix, error) {
	if err := p.Fit(X, y); err != nil {
		return nil, err
	}
	return p.Predict(X)
}

// FitTransform fits the pipeline and transforms the data.
// Equivalent to calling Fit followed by Transform.
func (p *Pipeline) FitTransform(X, y mat.Matrix) (mat.Matrix, error) {
	// Fit and transform all steps
	Xt := X
	var err error

	for _, step := range p.steps {
		// Check if step implements Transformer interface
		transformer, ok := step.Estimator.(model.Transformer)
		if !ok {
			return nil, errors.NewValidationError(
				"pipeline step",
				"all steps must be transformers for FitTransform",
				step.Name,
			)
		}

		// Fit the transformer (Transformer.Fit only takes X)
		if err = transformer.Fit(Xt); err != nil {
			return nil, errors.Wrap(err, fmt.Sprintf("failed to fit step '%s'", step.Name))
		}

		// Transform
		Xt, err = transformer.Transform(Xt)
		if err != nil {
			return nil, errors.Wrap(err, fmt.Sprintf("failed to transform at step '%s'", step.Name))
		}
	}

	p.state.SetFitted()
	return Xt, nil
}

// PredictProba applies transforms to the data, and predict_proba with the final estimator.
func (p *Pipeline) PredictProba(X mat.Matrix) (mat.Matrix, error) {
	if !p.state.IsFitted() {
		return nil, errors.NewNotFittedError("Pipeline", "PredictProba")
	}

	Xt, err := p.transform(X)
	if err != nil {
		return nil, err
	}

	// Predict probabilities with the final estimator
	if len(p.steps) > 0 {
		finalStep := p.steps[len(p.steps)-1]

		if predictor, ok := finalStep.Estimator.(interface {
			PredictProba(mat.Matrix) (mat.Matrix, error)
		}); ok {
			return predictor.PredictProba(Xt)
		}

		return nil, errors.NewValidationError(
			"pipeline final step",
			"final step must have PredictProba method",
			finalStep.Name,
		)
	}

	return nil, errors.New("pipeline has no steps")
}

// Score returns the score of the final estimator.
func (p *Pipeline) Score(X, y mat.Matrix) (float64, error) {
	if !p.state.IsFitted() {
		return 0, errors.NewNotFittedError("Pipeline", "Score")
	}

	Xt, err := p.transform(X)
	if err != nil {
		return 0, err
	}

	// Score with the final estimator
	if len(p.steps) > 0 {
		finalStep := p.steps[len(p.steps)-1]

		if scorer, ok := finalStep.Estimator.(interface {
			Score(mat.Matrix, mat.Matrix) (float64, error)
		}); ok {
			return scorer.Score(Xt, y)
		}

		return 0, errors.NewValidationError(
			"pipeline final step",
			"final step must have Score method",
			finalStep.Name,
		)
	}

	return 0, errors.New("pipeline has no steps")
}

// GetParams returns the parameters of the pipeline.
// This includes parameters of all steps.
func (p *Pipeline) GetParams() map[string]interface{} {
	params := make(map[string]interface{})
	params["steps"] = p.steps
	params["memory"] = p.memory
	params["verbose"] = p.verbose

	// Add parameters from each step
	for _, step := range p.steps {
		if paramsGetter, ok := step.Estimator.(interface {
			GetParams() map[string]interface{}
		}); ok {
			stepParams := paramsGetter.GetParams()
			for key, value := range stepParams {
				// Prefix with step name
				params[fmt.Sprintf("%s__%s", step.Name, key)] = value
			}
		}
	}

	return params
}

// SetParams sets the parameters of the pipeline.
func (p *Pipeline) SetParams(params map[string]interface{}) error {
	// Handle direct pipeline parameters
	if val, ok := params["verbose"]; ok {
		if verbose, ok := val.(bool); ok {
			p.verbose = verbose
		}
	}

	// TODO: Handle nested parameters for steps

	return nil
}

// NamedSteps returns the steps as a map for easy access by name.
func (p *Pipeline) NamedSteps() map[string]interface{} {
	return p.namedSteps_
}

// Steps returns the list of steps.
func (p *Pipeline) Steps() []Step {
	steps := make([]Step, len(p.steps))
	copy(steps, p.steps)
	return steps
}

// transform applies all transforms except the final estimator.
func (p *Pipeline) transform(X mat.Matrix) (mat.Matrix, error) {
	Xt := X
	var err error

	// Transform through all steps except the last
	for i := 0; i < len(p.steps)-1; i++ {
		step := p.steps[i]
		transformer, ok := step.Estimator.(model.Transformer)
		if !ok {
			return nil, errors.NewValidationError(
				"pipeline step",
				"intermediate steps must be transformers",
				step.Name,
			)
		}

		Xt, err = transformer.Transform(Xt)
		if err != nil {
			return nil, errors.Wrap(err, fmt.Sprintf("failed to transform at step '%s'", step.Name))
		}
	}

	return Xt, nil
}

// InverseTransform applies inverse transformations in reverse order.
// Only works if all steps are transformers with InverseTransform method.
func (p *Pipeline) InverseTransform(X mat.Matrix) (mat.Matrix, error) {
	if !p.state.IsFitted() {
		return nil, errors.NewNotFittedError("Pipeline", "InverseTransform")
	}

	Xt := X
	var err error

	// Apply inverse transforms in reverse order
	for i := len(p.steps) - 1; i >= 0; i-- {
		step := p.steps[i]

		inverseTransformer, ok := step.Estimator.(interface {
			InverseTransform(mat.Matrix) (mat.Matrix, error)
		})
		if !ok {
			return nil, errors.NewValidationError(
				"pipeline step",
				"all steps must have InverseTransform method",
				step.Name,
			)
		}

		Xt, err = inverseTransformer.InverseTransform(Xt)
		if err != nil {
			return nil, errors.Wrap(err, fmt.Sprintf("failed to inverse transform at step '%s'", step.Name))
		}
	}

	return Xt, nil
}
