package lightgbm

import (
	"fmt"
	"math"
	"time"

	"gonum.org/v1/gonum/mat"
)

// CallbackEnv contains the environment for callbacks
type CallbackEnv struct {
	Model        *Model
	Iteration    int
	BeginTime    time.Time
	EndTime      time.Time
	EvalResults  map[string]float64
	StopTraining bool
}

// Callback is a function that can be called during training
type Callback func(env *CallbackEnv) error

// PrintEvaluation prints evaluation results during training
func PrintEvaluation(period int) Callback {
	return func(env *CallbackEnv) error {
		if env.Iteration%period == 0 {
			fmt.Printf("[%d] ", env.Iteration)
			for name, value := range env.EvalResults {
				fmt.Printf("%s: %.6f ", name, value)
			}
			fmt.Println()
		}
		return nil
	}
}

// RecordEvaluation records evaluation history
func RecordEvaluation(history *map[string][]float64) Callback {
	return func(env *CallbackEnv) error {
		if *history == nil {
			*history = make(map[string][]float64)
		}
		for name, value := range env.EvalResults {
			(*history)[name] = append((*history)[name], value)
		}
		return nil
	}
}

// EarlyStoppingCallback implements early stopping as a callback
func EarlyStoppingCallback(rounds int, metric string, minimize bool) Callback {
	bestScore := math.Inf(1)
	if !minimize {
		bestScore = math.Inf(-1)
	}
	bestIteration := 0
	roundsNoImprove := 0

	return func(env *CallbackEnv) error {
		if value, exists := env.EvalResults[metric]; exists {
			improved := false
			if minimize {
				improved = value < bestScore
			} else {
				improved = value > bestScore
			}

			if improved {
				bestScore = value
				bestIteration = env.Iteration
				roundsNoImprove = 0
			} else {
				roundsNoImprove++
			}

			if roundsNoImprove >= rounds {
				fmt.Printf("Early stopping at iteration %d, best iteration was %d with %s = %.6f\n",
					env.Iteration, bestIteration, metric, bestScore)
				env.StopTraining = true
			}
		}
		return nil
	}
}

// TimeLimit stops training after a specified duration
func TimeLimit(maxDuration time.Duration) Callback {
	startTime := time.Now()
	return func(env *CallbackEnv) error {
		if time.Since(startTime) > maxDuration {
			fmt.Printf("Time limit reached at iteration %d\n", env.Iteration)
			env.StopTraining = true
		}
		return nil
	}
}

// ResetParameter allows dynamic parameter updates during training
func ResetParameter(paramFunc func(iteration int) map[string]interface{}) Callback {
	return func(env *CallbackEnv) error {
		params := paramFunc(env.Iteration)
		// Apply parameter updates to the model
		for key, value := range params {
			switch key {
			case "learning_rate":
				if lr, ok := value.(float64); ok {
					env.Model.LearningRate = lr
				}
			case "num_leaves":
				if nl, ok := value.(int); ok {
					env.Model.NumLeaves = nl
				}
				// Add more parameters as needed
			}
		}
		return nil
	}
}

// CallbackList manages multiple callbacks
type CallbackList struct {
	callbacks []Callback
	env       *CallbackEnv
}

// NewCallbackList creates a new callback list
func NewCallbackList(callbacks ...Callback) *CallbackList {
	return &CallbackList{
		callbacks: callbacks,
		env: &CallbackEnv{
			EvalResults: make(map[string]float64),
		},
	}
}

// BeforeIteration calls callbacks before each iteration
func (cl *CallbackList) BeforeIteration(iteration int, model *Model) error {
	cl.env.Iteration = iteration
	cl.env.Model = model
	cl.env.BeginTime = time.Now()

	for _, cb := range cl.callbacks {
		if err := cb(cl.env); err != nil {
			return err
		}
		if cl.env.StopTraining {
			break
		}
	}
	return nil
}

// AfterIteration calls callbacks after each iteration
func (cl *CallbackList) AfterIteration(iteration int, model *Model, evalResults map[string]float64) error {
	cl.env.Iteration = iteration
	cl.env.Model = model
	cl.env.EndTime = time.Now()
	cl.env.EvalResults = evalResults

	for _, cb := range cl.callbacks {
		if err := cb(cl.env); err != nil {
			return err
		}
	}
	return nil
}

// ShouldStop returns whether training should stop
func (cl *CallbackList) ShouldStop() bool {
	return cl.env.StopTraining
}

// LearningRateSchedule implements learning rate decay
func LearningRateSchedule(decayRate float64, decaySteps int) Callback {
	initialLR := -1.0
	return func(env *CallbackEnv) error {
		if initialLR < 0 {
			initialLR = env.Model.LearningRate
		}

		if env.Iteration > 0 && env.Iteration%decaySteps == 0 {
			env.Model.LearningRate *= decayRate
			fmt.Printf("Learning rate updated to %.6f at iteration %d\n",
				env.Model.LearningRate, env.Iteration)
		}
		return nil
	}
}

// ModelCheckpoint saves the model at specified intervals
func ModelCheckpoint(filepath string, period int, saveOnlyImproved bool) Callback {
	bestScore := math.Inf(1)
	return func(env *CallbackEnv) error {
		if env.Iteration%period != 0 {
			return nil
		}

		save := true
		if saveOnlyImproved {
			// Check if model improved (using first metric)
			for _, value := range env.EvalResults {
				if value < bestScore {
					bestScore = value
				} else {
					save = false
				}
				break
			}
		}

		if save {
			filename := fmt.Sprintf("%s_iter_%d.json", filepath, env.Iteration)
			if err := env.Model.SaveToJSON(filename); err != nil {
				return fmt.Errorf("failed to save checkpoint: %w", err)
			}
			fmt.Printf("Model checkpoint saved to %s\n", filename)
		}
		return nil
	}
}

// FitWithCallbacks trains the model with callbacks
func (t *Trainer) FitWithCallbacks(X, y mat.Matrix, callbacks ...Callback) error {
	// Set callbacks
	t.callbacks = NewCallbackList(callbacks...)

	// Regular fit
	if err := t.Fit(X, y); err != nil {
		return err
	}

	return nil
}
