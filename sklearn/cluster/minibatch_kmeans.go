package cluster

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/YuminosukeSato/scigo/core/model"
	"github.com/YuminosukeSato/scigo/pkg/errors"
	"gonum.org/v1/gonum/mat"
	"github.com/YuminosukeSato/scigo/pkg/log"
)

var globalProvider log.LoggerProvider

// MiniBatchKMeans implements mini-batch K-means clustering
// Compatible with scikit-learn's MiniBatchKMeans
type MiniBatchKMeans struct {
	// State management using composition
	state  *model.StateManager
	logger log.Logger

	// Hyperparameters
	nClusters         int     // Number of clusters
	init              string  // Initialization method: "k-means++", "random"
	maxIter           int     // Maximum number of iterations
	batchSize         int     // Mini-batch size
	verbose           int     // Verbosity level
	computeLabels     bool    // Whether to compute labels
	randomState       int64   // Random seed
	tol               float64 // Tolerance for convergence
	maxNoImprovement  int     // Maximum iterations without improvement
	initSize          int     // Number of samples for initialization
	nInit             int     // Number of runs with different initializations
	reassignmentRatio float64 // Reassignment ratio

	// Learning parameters
	clusterCenters_ [][]float64 // Cluster centers (nClusters x nFeatures)
	labels_         []int       // Cluster label for each sample
	inertia_        float64     // Within-cluster sum of squared errors
	nIter_          int         // Number of iterations executed
	counts_         []int       // Number of samples per cluster

	// Internal state
	mu         sync.RWMutex
	rng        *rand.Rand
	nFeatures_ int
	nSamples_  int
}

// NewMiniBatchKMeans creates a new MiniBatchKMeans instance
func NewMiniBatchKMeans(options ...KMeansOption) *MiniBatchKMeans {
	kmeans := &MiniBatchKMeans{
		nClusters:         8,
		init:              "k-means++",
		maxIter:           100,
		batchSize:         100,
		verbose:           0,
		computeLabels:     true,
		randomState:       -1,
		tol:               0.0,
		maxNoImprovement:  10,
		initSize:          -1, // Default is 3 * batchSize
		nInit:             3,
		reassignmentRatio: 0.01,
	}

	for _, opt := range options {
		opt(kmeans)
	}

	// Adjust default values
	if kmeans.initSize == -1 {
		kmeans.initSize = 3 * kmeans.batchSize
	}

	if kmeans.randomState >= 0 {
		kmeans.rng = rand.New(rand.NewSource(kmeans.randomState))
	} else {
		kmeans.rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	// Initialize state manager and logger
	kmeans.state = model.NewStateManager()
	// Initialize logger using global provider
	if globalProvider == nil {
		globalProvider = log.NewZerologProvider(log.ToLogLevel("info"))
	}
	kmeans.logger = globalProvider.GetLoggerWithName("MiniBatchKMeans")

	return kmeans
}

// KMeansOption is a configuration option for MiniBatchKMeans
type KMeansOption func(*MiniBatchKMeans)

// WithKMeansNClusters sets the number of clusters
func WithKMeansNClusters(n int) KMeansOption {
	return func(kmeans *MiniBatchKMeans) {
		kmeans.nClusters = n
	}
}

// WithKMeansInit sets the initialization method
func WithKMeansInit(init string) KMeansOption {
	return func(kmeans *MiniBatchKMeans) {
		kmeans.init = init
	}
}

// WithKMeansMaxIter sets the maximum number of iterations
func WithKMeansMaxIter(maxIter int) KMeansOption {
	return func(kmeans *MiniBatchKMeans) {
		kmeans.maxIter = maxIter
	}
}

// WithKMeansBatchSize sets the mini-batch size
func WithKMeansBatchSize(batchSize int) KMeansOption {
	return func(kmeans *MiniBatchKMeans) {
		kmeans.batchSize = batchSize
	}
}

// WithKMeansRandomState sets the random seed
func WithKMeansRandomState(seed int64) KMeansOption {
	return func(kmeans *MiniBatchKMeans) {
		kmeans.randomState = seed
		if seed >= 0 {
			kmeans.rng = rand.New(rand.NewSource(seed))
		}
	}
}

// WithKMeansTol sets the tolerance for convergence
func WithKMeansTol(tol float64) KMeansOption {
	return func(kmeans *MiniBatchKMeans) {
		kmeans.tol = tol
	}
}

// Fit trains the model using batch learning
func (kmeans *MiniBatchKMeans) Fit(X, y mat.Matrix) error {
	kmeans.mu.Lock()
	defer kmeans.mu.Unlock()

	rows, cols := X.Dims()
	kmeans.nSamples_ = rows
	kmeans.nFeatures_ = cols

	if rows < kmeans.nClusters {
		return errors.Newf("number of samples is less than number of clusters: %d < %d", rows, kmeans.nClusters)
	}

	// Run multiple times and select the best result
	bestInertia := math.Inf(1)
	var bestCenters [][]float64
	var bestLabels []int
	var bestNIter int

	for run := 0; run < kmeans.nInit; run++ {
		centers, labels, inertia, nIter := kmeans.fitSingleRun(X)

		if inertia < bestInertia {
			bestInertia = inertia
			bestCenters = centers
			bestLabels = labels
			bestNIter = nIter
		}
	}

	kmeans.clusterCenters_ = bestCenters
	kmeans.labels_ = bestLabels
	kmeans.inertia_ = bestInertia
	kmeans.nIter_ = bestNIter

	kmeans.state.SetFitted()
	return nil
}

// fitSingleRun performs a single training run
func (kmeans *MiniBatchKMeans) fitSingleRun(X mat.Matrix) ([][]float64, []int, float64, int) {
	rows, cols := X.Dims()

	// Initialize cluster centers
	centers := kmeans.initializeCenters(X)
	counts := make([]int, kmeans.nClusters)

	prevInertia := math.Inf(1)
	noImprovementCount := 0
	var finalIter int

	for iter := 0; iter < kmeans.maxIter; iter++ {
		finalIter = iter
		// Select mini-batch
		batchIndices := kmeans.selectMiniBatch(rows)

		// Assign each mini-batch sample to nearest cluster
		for _, idx := range batchIndices {
			sample := mat.Row(nil, idx, X)
			nearestCluster := kmeans.findNearestCluster(sample, centers)

			// Update cluster centers
			counts[nearestCluster]++
			eta := 1.0 / float64(counts[nearestCluster])

			for j := 0; j < cols; j++ {
				centers[nearestCluster][j] = (1-eta)*centers[nearestCluster][j] + eta*sample[j]
			}
		}

		// Calculate inertia
		inertia := kmeans.computeInertia(X, centers)

		// Check convergence
		if prevInertia-inertia < kmeans.tol {
			noImprovementCount++
			if noImprovementCount >= kmeans.maxNoImprovement {
				break
			}
		} else {
			noImprovementCount = 0
		}

		prevInertia = inertia

		if kmeans.verbose > 0 && iter%10 == 0 {
			fmt.Printf("Iteration %d, inertia: %.6f\n", iter, inertia)
		}
	}

	// Calculate final labels
	var labels []int
	if kmeans.computeLabels {
		labels = make([]int, rows)
		for i := 0; i < rows; i++ {
			sample := mat.Row(nil, i, X)
			labels[i] = kmeans.findNearestCluster(sample, centers)
		}
	}

	finalInertia := kmeans.computeInertia(X, centers)
	return centers, labels, finalInertia, finalIter
}

// PartialFit trains the model incrementally with mini-batches
func (kmeans *MiniBatchKMeans) PartialFit(X, y mat.Matrix, classes []int) error {
	kmeans.mu.Lock()
	defer kmeans.mu.Unlock()

	rows, cols := X.Dims()

	// Initialize on first call
	if kmeans.clusterCenters_ == nil {
		kmeans.nFeatures_ = cols
		kmeans.clusterCenters_ = kmeans.initializeCenters(X)
		kmeans.counts_ = make([]int, kmeans.nClusters)
	}

	if cols != kmeans.nFeatures_ {
		return errors.NewDimensionError("PartialFit", kmeans.nFeatures_, cols, 1)
	}

	// Process mini-batch
	for i := 0; i < rows; i++ {
		sample := mat.Row(nil, i, X)
		nearestCluster := kmeans.findNearestCluster(sample, kmeans.clusterCenters_)

		// Update cluster centers
		kmeans.counts_[nearestCluster]++
		eta := 1.0 / float64(kmeans.counts_[nearestCluster])

		for j := 0; j < cols; j++ {
			kmeans.clusterCenters_[nearestCluster][j] =
				(1-eta)*kmeans.clusterCenters_[nearestCluster][j] + eta*sample[j]
		}
	}

	kmeans.state.SetFitted()
	return nil
}

// Transform converts data to distances to cluster centers
func (kmeans *MiniBatchKMeans) Transform(X mat.Matrix) (mat.Matrix, error) {
	kmeans.mu.RLock()
	defer kmeans.mu.RUnlock()

	if !kmeans.state.IsFitted() {
		return nil, errors.New("model is not fitted")
	}

	rows, cols := X.Dims()
	if cols != kmeans.nFeatures_ {
		return nil, errors.Newf("feature dimension mismatch: expected %d, got %d", kmeans.nFeatures_, cols)
	}

	distances := mat.NewDense(rows, kmeans.nClusters, nil)

	for i := 0; i < rows; i++ {
		sample := mat.Row(nil, i, X)
		for c := 0; c < kmeans.nClusters; c++ {
			dist := euclideanDistance(sample, kmeans.clusterCenters_[c])
			distances.Set(i, c, dist)
		}
	}

	return distances, nil
}

// Predict performs cluster prediction on input data
func (kmeans *MiniBatchKMeans) Predict(X mat.Matrix) (mat.Matrix, error) {
	kmeans.mu.RLock()
	defer kmeans.mu.RUnlock()

	if !kmeans.state.IsFitted() {
		return nil, errors.New("model is not fitted")
	}

	rows, cols := X.Dims()
	if cols != kmeans.nFeatures_ {
		return nil, errors.Newf("feature dimension mismatch: expected %d, got %d", kmeans.nFeatures_, cols)
	}

	predictions := mat.NewDense(rows, 1, nil)

	for i := 0; i < rows; i++ {
		sample := mat.Row(nil, i, X)
		cluster := kmeans.findNearestCluster(sample, kmeans.clusterCenters_)
		predictions.Set(i, 0, float64(cluster))
	}

	return predictions, nil
}

// FitPredict performs training and prediction simultaneously
func (kmeans *MiniBatchKMeans) FitPredict(X, y mat.Matrix) (mat.Matrix, error) {
	err := kmeans.Fit(X, y)
	if err != nil {
		return nil, err
	}
	return kmeans.Predict(X)
}

// Streaming learning methods

// FitStream trains the model from data stream
func (kmeans *MiniBatchKMeans) FitStream(ctx context.Context, dataChan <-chan *model.Batch) error {
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case batch, ok := <-dataChan:
			if !ok {
				return nil
			}
			if err := kmeans.PartialFit(batch.X, batch.Y, nil); err != nil {
				return err
			}
		}
	}
}

// PredictStream performs real-time prediction on input stream
func (kmeans *MiniBatchKMeans) PredictStream(ctx context.Context, inputChan <-chan mat.Matrix) <-chan mat.Matrix {
	outputChan := make(chan mat.Matrix)

	go func() {
		defer close(outputChan)

		for {
			select {
			case <-ctx.Done():
				return
			case X, ok := <-inputChan:
				if !ok {
					return
				}

				pred, err := kmeans.Predict(X)
				if err != nil {
					continue
				}

				select {
				case outputChan <- pred:
				case <-ctx.Done():
					return
				}
			}
		}
	}()

	return outputChan
}

// Interface implementation methods

// NIterations returns the number of training iterations performed
func (kmeans *MiniBatchKMeans) NIterations() int {
	kmeans.mu.RLock()
	defer kmeans.mu.RUnlock()
	return kmeans.nIter_
}

// IsWarmStart returns whether warm start is enabled (always false)
func (kmeans *MiniBatchKMeans) IsWarmStart() bool {
	return false
}

// SetWarmStart sets warm start enable/disable (does nothing)
func (kmeans *MiniBatchKMeans) SetWarmStart(warmStart bool) {
	// MiniBatchKMeans does not support warm start
}

// ClusterCenters returns the learned cluster centers
func (kmeans *MiniBatchKMeans) ClusterCenters() [][]float64 {
	kmeans.mu.RLock()
	defer kmeans.mu.RUnlock()

	centers := make([][]float64, len(kmeans.clusterCenters_))
	for i := range kmeans.clusterCenters_ {
		centers[i] = make([]float64, len(kmeans.clusterCenters_[i]))
		copy(centers[i], kmeans.clusterCenters_[i])
	}
	return centers
}

// Labels returns cluster labels for training data
func (kmeans *MiniBatchKMeans) Labels() []int {
	kmeans.mu.RLock()
	defer kmeans.mu.RUnlock()

	if kmeans.labels_ == nil {
		return nil
	}

	labels := make([]int, len(kmeans.labels_))
	copy(labels, kmeans.labels_)
	return labels
}

// Inertia returns inertia (within-cluster sum of squared errors)
func (kmeans *MiniBatchKMeans) Inertia() float64 {
	kmeans.mu.RLock()
	defer kmeans.mu.RUnlock()
	return kmeans.inertia_
}

// Internal helper methods

// initializeCenters initializes cluster centers
func (kmeans *MiniBatchKMeans) initializeCenters(X mat.Matrix) [][]float64 {
	rows, cols := X.Dims()
	centers := make([][]float64, kmeans.nClusters)

	switch kmeans.init {
	case "k-means++":
		return kmeans.initKMeansPlusPlus(X)
	case "random":
		for i := 0; i < kmeans.nClusters; i++ {
			centers[i] = make([]float64, cols)
			idx := kmeans.rng.Intn(rows)
			sample := mat.Row(nil, idx, X)
			copy(centers[i], sample)
		}
	default:
		// Default is k-means++
		return kmeans.initKMeansPlusPlus(X)
	}

	return centers
}

// initKMeansPlusPlus performs k-means++ initialization
func (kmeans *MiniBatchKMeans) initKMeansPlusPlus(X mat.Matrix) [][]float64 {
	rows, cols := X.Dims()
	centers := make([][]float64, kmeans.nClusters)

	// Randomly select the first cluster center
	centers[0] = make([]float64, cols)
	idx := kmeans.rng.Intn(rows)
	sample := mat.Row(nil, idx, X)
	copy(centers[0], sample)

	// Select remaining cluster centers
	for c := 1; c < kmeans.nClusters; c++ {
		distances := make([]float64, rows)
		totalDistance := 0.0

		// Calculate squared distance from each sample to nearest cluster center
		for i := 0; i < rows; i++ {
			sample := mat.Row(nil, i, X)
			minDist := math.Inf(1)

			for j := 0; j < c; j++ {
				dist := euclideanDistance(sample, centers[j])
				if dist < minDist {
					minDist = dist
				}
			}

			distances[i] = minDist * minDist
			totalDistance += distances[i]
		}

		// Select sample according to probability
		target := kmeans.rng.Float64() * totalDistance
		cumSum := 0.0
		selectedIdx := 0

		for i := 0; i < rows; i++ {
			cumSum += distances[i]
			if cumSum >= target {
				selectedIdx = i
				break
			}
		}

		centers[c] = make([]float64, cols)
		sample = mat.Row(nil, selectedIdx, X)
		copy(centers[c], sample)
	}

	return centers
}

// selectMiniBatch selects sample indices for mini-batch
func (kmeans *MiniBatchKMeans) selectMiniBatch(nSamples int) []int {
	batchSize := kmeans.batchSize
	if batchSize > nSamples {
		batchSize = nSamples
	}

	// Random sampling
	indices := make([]int, nSamples)
	for i := range indices {
		indices[i] = i
	}

	// Fisher-Yates shuffle
	for i := nSamples - 1; i > 0; i-- {
		j := kmeans.rng.Intn(i + 1)
		indices[i], indices[j] = indices[j], indices[i]
	}

	return indices[:batchSize]
}

// findNearestCluster searches for the nearest cluster
func (kmeans *MiniBatchKMeans) findNearestCluster(sample []float64, centers [][]float64) int {
	minDist := math.Inf(1)
	nearestCluster := 0

	for c, center := range centers {
		dist := euclideanDistance(sample, center)
		if dist < minDist {
			minDist = dist
			nearestCluster = c
		}
	}

	return nearestCluster
}

// computeInertia calculates inertia (within-cluster sum of squared errors)
func (kmeans *MiniBatchKMeans) computeInertia(X mat.Matrix, centers [][]float64) float64 {
	rows, _ := X.Dims()
	inertia := 0.0

	for i := 0; i < rows; i++ {
		sample := mat.Row(nil, i, X)
		nearestCluster := kmeans.findNearestCluster(sample, centers)
		dist := euclideanDistance(sample, centers[nearestCluster])
		inertia += dist * dist
	}

	return inertia
}

// Utility functions

// euclideanDistance calculates Euclidean distance
func euclideanDistance(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}

	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return math.Sqrt(sum)
}
