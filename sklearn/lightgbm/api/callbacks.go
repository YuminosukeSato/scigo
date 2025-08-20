package api

import (
	"errors"
	"fmt"
	"math"
)

// ErrEarlyStop is returned when early stopping is triggered
var ErrEarlyStop = errors.New("early stopping")

// Callback is an interface for training callbacks
type Callback interface {
	Init(env *CallbackEnv) error
	BeforeIteration(env *CallbackEnv) error
	AfterIteration(env *CallbackEnv) error
	Finalize(env *CallbackEnv) error
}

// CallbackEnv holds the environment for callbacks
type CallbackEnv struct {
	Iteration     int
	NumBoostRound int
	Booster       *Booster
	Params        map[string]interface{}
	EvalResults   map[string][]float64
	BestIteration int
	BestScore     float64
	ShouldStop    bool
}

// PrintEvaluationCallback prints evaluation results during training
// Similar to Python's verbose_eval callback
type PrintEvaluationCallback struct {
	period   int
	showStd  bool
	lastTime int
}

// NewPrintEvaluationCallback creates a new print evaluation callback
func NewPrintEvaluationCallback(period int) *PrintEvaluationCallback {
	return &PrintEvaluationCallback{
		period:   period,
		showStd:  false,
		lastTime: -1,
	}
}

// Init initializes the callback
func (cb *PrintEvaluationCallback) Init(_ *CallbackEnv) error {
	if cb.period <= 0 {
		cb.period = 1
	}
	return nil
}

// BeforeIteration is called before each iteration
func (cb *PrintEvaluationCallback) BeforeIteration(_ *CallbackEnv) error {
	return nil
}

// AfterIteration is called after each iteration
func (cb *PrintEvaluationCallback) AfterIteration(env *CallbackEnv) error {
	if env.Iteration%cb.period != 0 {
		return nil
	}

	// Format: [Iteration] train-metric: value valid-metric: value
	output := fmt.Sprintf("[%d]", env.Iteration+1)

	// Print all evaluation results
	for key, values := range env.EvalResults {
		if len(values) > 0 {
			lastValue := values[len(values)-1]
			output += fmt.Sprintf("\t%s: %.6f", key, lastValue)
		}
	}

	if output != fmt.Sprintf("[%d]", env.Iteration+1) {
		fmt.Println(output)
	}

	cb.lastTime = env.Iteration
	return nil
}

// Finalize is called after training
func (cb *PrintEvaluationCallback) Finalize(env *CallbackEnv) error {
	// Print final results if not already printed
	if cb.lastTime < env.Iteration-1 && env.Iteration > 0 {
		output := fmt.Sprintf("[%d]", env.Iteration)
		for key, values := range env.EvalResults {
			if len(values) > 0 {
				lastValue := values[len(values)-1]
				output += fmt.Sprintf("\t%s: %.6f", key, lastValue)
			}
		}
		fmt.Println(output)
	}
	return nil
}

// EarlyStoppingCallback implements early stopping
// Similar to Python's early_stopping callback
type EarlyStoppingCallback struct {
	stoppingRounds int
	verbose        bool
	firstMetric    bool
	minDelta       float64

	// Internal state
	bestScore      float64
	bestIteration  int
	waitCount      int
	isHigherBetter bool
	metricName     string
}

// NewEarlyStoppingCallback creates a new early stopping callback
func NewEarlyStoppingCallback(stoppingRounds int, verbose bool) *EarlyStoppingCallback {
	return &EarlyStoppingCallback{
		stoppingRounds: stoppingRounds,
		verbose:        verbose,
		firstMetric:    true,
		minDelta:       0.0,
		bestScore:      math.Inf(1),
		bestIteration:  0,
		waitCount:      0,
		isHigherBetter: false,
	}
}

// Init initializes the callback
func (cb *EarlyStoppingCallback) Init(env *CallbackEnv) error {
	// Determine metric direction based on objective
	if objective, ok := env.Params["objective"].(string); ok {
		switch objective {
		case "binary", "multiclass":
			// For classification, higher accuracy is better
			cb.isHigherBetter = true
			cb.bestScore = math.Inf(-1)
		default:
			// For regression, lower error is better
			cb.isHigherBetter = false
			cb.bestScore = math.Inf(1)
		}
	}

	if cb.verbose {
		fmt.Printf("[LightGBM] [Info] Early stopping is enabled. Will stop if metric doesn't improve for %d rounds.\n",
			cb.stoppingRounds)
	}

	return nil
}

// BeforeIteration is called before each iteration
func (cb *EarlyStoppingCallback) BeforeIteration(_ *CallbackEnv) error {
	return nil
}

// AfterIteration is called after each iteration
func (cb *EarlyStoppingCallback) AfterIteration(env *CallbackEnv) error {
	// Get the first validation metric
	var currentScore float64
	var metricKey string

	for key, values := range env.EvalResults {
		if len(values) > 0 {
			// Use the first validation metric found
			metricKey = key
			currentScore = values[len(values)-1]
			break
		}
	}

	if metricKey == "" {
		// No validation metric, skip early stopping
		return nil
	}

	if cb.metricName == "" {
		cb.metricName = metricKey
	}

	// Check if score improved
	improved := false
	if cb.isHigherBetter {
		if currentScore > cb.bestScore+cb.minDelta {
			improved = true
		}
	} else {
		if currentScore < cb.bestScore-cb.minDelta {
			improved = true
		}
	}

	if improved {
		cb.bestScore = currentScore
		cb.bestIteration = env.Iteration
		cb.waitCount = 0
		env.BestIteration = env.Iteration
		env.BestScore = currentScore

		if cb.verbose {
			fmt.Printf("[%d]\tBest iteration so far, %s: %.6f\n",
				env.Iteration+1, cb.metricName, cb.bestScore)
		}
	} else {
		cb.waitCount++

		if cb.waitCount >= cb.stoppingRounds {
			if cb.verbose {
				fmt.Printf("[LightGBM] [Info] Early stopping at iteration %d, best iteration is %d\n",
					env.Iteration+1, cb.bestIteration+1)
				fmt.Printf("[LightGBM] [Info] Best %s: %.6f\n", cb.metricName, cb.bestScore)
			}

			env.ShouldStop = true
			env.Booster.SetBestIteration(cb.bestIteration)
			return ErrEarlyStop
		}
	}

	return nil
}

// Finalize is called after training
func (cb *EarlyStoppingCallback) Finalize(_ *CallbackEnv) error {
	if cb.verbose && cb.bestIteration > 0 {
		fmt.Printf("[LightGBM] [Info] Best iteration: %d\n", cb.bestIteration+1)
	}
	return nil
}

// RecordEvaluationCallback records evaluation results
type RecordEvaluationCallback struct {
	evalResult map[string]map[string][]float64
}

// NewRecordEvaluationCallback creates a new record evaluation callback
func NewRecordEvaluationCallback() *RecordEvaluationCallback {
	return &RecordEvaluationCallback{
		evalResult: make(map[string]map[string][]float64),
	}
}

// Init initializes the callback
func (cb *RecordEvaluationCallback) Init(env *CallbackEnv) error {
	return nil
}

// BeforeIteration is called before each iteration
func (cb *RecordEvaluationCallback) BeforeIteration(env *CallbackEnv) error {
	return nil
}

// AfterIteration is called after each iteration
func (cb *RecordEvaluationCallback) AfterIteration(env *CallbackEnv) error {
	// Record all evaluation results
	for key, values := range env.EvalResults {
		// Parse dataset and metric from key (format: "dataset-metric")
		var dataset, metric string
		for i := len(key) - 1; i >= 0; i-- {
			if key[i] == '-' {
				dataset = key[:i]
				metric = key[i+1:]
				break
			}
		}

		if dataset == "" || metric == "" {
			continue
		}

		if _, ok := cb.evalResult[dataset]; !ok {
			cb.evalResult[dataset] = make(map[string][]float64)
		}

		cb.evalResult[dataset][metric] = values
	}

	return nil
}

// Finalize is called after training
func (cb *RecordEvaluationCallback) Finalize(env *CallbackEnv) error {
	return nil
}

// GetEvalResult returns the recorded evaluation results
func (cb *RecordEvaluationCallback) GetEvalResult() map[string]map[string][]float64 {
	return cb.evalResult
}

// ResetParameterCallback resets parameters during training
type ResetParameterCallback struct {
	startIteration int
	newParams      map[string]interface{}
}

// NewResetParameterCallback creates a new reset parameter callback
func NewResetParameterCallback(startIteration int, newParams map[string]interface{}) *ResetParameterCallback {
	return &ResetParameterCallback{
		startIteration: startIteration,
		newParams:      newParams,
	}
}

// Init initializes the callback
func (cb *ResetParameterCallback) Init(env *CallbackEnv) error {
	return nil
}

// BeforeIteration is called before each iteration
func (cb *ResetParameterCallback) BeforeIteration(env *CallbackEnv) error {
	if env.Iteration == cb.startIteration {
		// Update parameters
		for key, value := range cb.newParams {
			env.Params[key] = value
		}
		fmt.Printf("[LightGBM] [Info] Parameters reset at iteration %d\n", env.Iteration+1)
	}
	return nil
}

// AfterIteration is called after each iteration
func (cb *ResetParameterCallback) AfterIteration(env *CallbackEnv) error {
	return nil
}

// Finalize is called after training
func (cb *ResetParameterCallback) Finalize(env *CallbackEnv) error {
	return nil
}

// CallbackList manages multiple callbacks
type CallbackList struct {
	callbacks []Callback
}

// NewCallbackList creates a new callback list
func NewCallbackList(callbacks ...Callback) *CallbackList {
	return &CallbackList{
		callbacks: callbacks,
	}
}

// Add adds a callback to the list
func (cl *CallbackList) Add(callback Callback) {
	cl.callbacks = append(cl.callbacks, callback)
}

// Init initializes all callbacks
func (cl *CallbackList) Init(env *CallbackEnv) error {
	for _, cb := range cl.callbacks {
		if err := cb.Init(env); err != nil {
			return err
		}
	}
	return nil
}

// BeforeIteration calls BeforeIteration on all callbacks
func (cl *CallbackList) BeforeIteration(env *CallbackEnv) error {
	for _, cb := range cl.callbacks {
		if err := cb.BeforeIteration(env); err != nil {
			return err
		}
	}
	return nil
}

// AfterIteration calls AfterIteration on all callbacks
func (cl *CallbackList) AfterIteration(env *CallbackEnv) error {
	for _, cb := range cl.callbacks {
		if err := cb.AfterIteration(env); err != nil {
			return err
		}
	}
	return nil
}

// Finalize calls Finalize on all callbacks
func (cl *CallbackList) Finalize(env *CallbackEnv) error {
	for _, cb := range cl.callbacks {
		if err := cb.Finalize(env); err != nil {
			return err
		}
	}
	return nil
}
