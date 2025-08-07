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
)

// MiniBatchKMeans はミニバッチK-meansクラスタリング
// scikit-learnのMiniBatchKMeansと互換性を持つ
type MiniBatchKMeans struct {
	model.BaseEstimator

	// ハイパーパラメータ
	nClusters         int     // クラスタ数
	init              string  // 初期化方法: "k-means++", "random"
	maxIter           int     // 最大イテレーション数
	batchSize         int     // ミニバッチサイズ
	verbose           int     // 詳細出力レベル
	computeLabels     bool    // ラベルを計算するか
	randomState       int64   // 乱数シード
	tol               float64 // 収束判定の許容誤差
	maxNoImprovement  int     // 改善なしの最大イテレーション数
	initSize          int     // 初期化用のサンプル数
	nInit             int     // 異なる初期化での実行回数
	reassignmentRatio float64 // 再割り当て比率

	// 学習パラメータ
	clusterCenters_ [][]float64 // クラスタ中心（nClusters x nFeatures）
	labels_         []int       // 各サンプルのクラスタラベル
	inertia_        float64     // クラスタ内平方和誤差
	nIter_          int         // 実行されたイテレーション数
	counts_         []int       // 各クラスタのサンプル数

	// 内部状態
	mu         sync.RWMutex
	rng        *rand.Rand
	nFeatures_ int
	nSamples_  int
}

// NewMiniBatchKMeans は新しいMiniBatchKMeansを作成
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
		initSize:          -1, // デフォルトは3 * batchSize
		nInit:             3,
		reassignmentRatio: 0.01,
	}

	for _, opt := range options {
		opt(kmeans)
	}

	// デフォルト値の調整
	if kmeans.initSize == -1 {
		kmeans.initSize = 3 * kmeans.batchSize
	}

	if kmeans.randomState >= 0 {
		kmeans.rng = rand.New(rand.NewSource(kmeans.randomState))
	} else {
		kmeans.rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	return kmeans
}

// KMeansOption はMiniBatchKMeansの設定オプション
type KMeansOption func(*MiniBatchKMeans)

// WithKMeansNClusters はクラスタ数を設定
func WithKMeansNClusters(n int) KMeansOption {
	return func(kmeans *MiniBatchKMeans) {
		kmeans.nClusters = n
	}
}

// WithKMeansInit は初期化方法を設定
func WithKMeansInit(init string) KMeansOption {
	return func(kmeans *MiniBatchKMeans) {
		kmeans.init = init
	}
}

// WithKMeansMaxIter は最大イテレーション数を設定
func WithKMeansMaxIter(maxIter int) KMeansOption {
	return func(kmeans *MiniBatchKMeans) {
		kmeans.maxIter = maxIter
	}
}

// WithKMeansBatchSize はミニバッチサイズを設定
func WithKMeansBatchSize(batchSize int) KMeansOption {
	return func(kmeans *MiniBatchKMeans) {
		kmeans.batchSize = batchSize
	}
}

// WithKMeansRandomState は乱数シードを設定
func WithKMeansRandomState(seed int64) KMeansOption {
	return func(kmeans *MiniBatchKMeans) {
		kmeans.randomState = seed
		if seed >= 0 {
			kmeans.rng = rand.New(rand.NewSource(seed))
		}
	}
}

// WithKMeansTol は収束判定の許容誤差を設定
func WithKMeansTol(tol float64) KMeansOption {
	return func(kmeans *MiniBatchKMeans) {
		kmeans.tol = tol
	}
}

// Fit はバッチ学習でモデルを訓練
func (kmeans *MiniBatchKMeans) Fit(X, y mat.Matrix) error {
	kmeans.mu.Lock()
	defer kmeans.mu.Unlock()

	rows, cols := X.Dims()
	kmeans.nSamples_ = rows
	kmeans.nFeatures_ = cols

	if rows < kmeans.nClusters {
		return errors.Newf("サンプル数がクラスタ数より少ないです: %d < %d", rows, kmeans.nClusters)
	}

	// 複数回実行して最良の結果を選択
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

	kmeans.SetFitted()
	return nil
}

// fitSingleRun は単一回の学習を実行
func (kmeans *MiniBatchKMeans) fitSingleRun(X mat.Matrix) ([][]float64, []int, float64, int) {
	rows, cols := X.Dims()

	// クラスタ中心の初期化
	centers := kmeans.initializeCenters(X)
	counts := make([]int, kmeans.nClusters)

	prevInertia := math.Inf(1)
	noImprovementCount := 0
	var finalIter int

	for iter := 0; iter < kmeans.maxIter; iter++ {
		finalIter = iter
		// ミニバッチの選択
		batchIndices := kmeans.selectMiniBatch(rows)

		// 各ミニバッチサンプルを最近傍クラスタに割り当て
		for _, idx := range batchIndices {
			sample := mat.Row(nil, idx, X)
			nearestCluster := kmeans.findNearestCluster(sample, centers)

			// クラスタ中心の更新
			counts[nearestCluster]++
			eta := 1.0 / float64(counts[nearestCluster])

			for j := 0; j < cols; j++ {
				centers[nearestCluster][j] = (1-eta)*centers[nearestCluster][j] + eta*sample[j]
			}
		}

		// 慣性（inertia）の計算
		inertia := kmeans.computeInertia(X, centers)

		// 収束判定
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

	// 最終的なラベルの計算
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

// PartialFit はミニバッチでモデルを逐次的に学習
func (kmeans *MiniBatchKMeans) PartialFit(X, y mat.Matrix, classes []int) error {
	kmeans.mu.Lock()
	defer kmeans.mu.Unlock()

	rows, cols := X.Dims()

	// 初回呼び出し時の初期化
	if kmeans.clusterCenters_ == nil {
		kmeans.nFeatures_ = cols
		kmeans.clusterCenters_ = kmeans.initializeCenters(X)
		kmeans.counts_ = make([]int, kmeans.nClusters)
	}

	if cols != kmeans.nFeatures_ {
		return errors.NewDimensionError("PartialFit", kmeans.nFeatures_, cols, 1)
	}

	// ミニバッチ処理
	for i := 0; i < rows; i++ {
		sample := mat.Row(nil, i, X)
		nearestCluster := kmeans.findNearestCluster(sample, kmeans.clusterCenters_)

		// クラスタ中心の更新
		kmeans.counts_[nearestCluster]++
		eta := 1.0 / float64(kmeans.counts_[nearestCluster])

		for j := 0; j < cols; j++ {
			kmeans.clusterCenters_[nearestCluster][j] =
				(1-eta)*kmeans.clusterCenters_[nearestCluster][j] + eta*sample[j]
		}
	}

	kmeans.SetFitted()
	return nil
}

// Transform はデータをクラスタ中心との距離に変換
func (kmeans *MiniBatchKMeans) Transform(X mat.Matrix) (mat.Matrix, error) {
	kmeans.mu.RLock()
	defer kmeans.mu.RUnlock()

	if !kmeans.IsFitted() {
		return nil, errors.New("モデルが学習されていません")
	}

	rows, cols := X.Dims()
	if cols != kmeans.nFeatures_ {
		return nil, errors.Newf("特徴量の次元が一致しません: expected %d, got %d", kmeans.nFeatures_, cols)
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

// Predict は入力データに対するクラスタ予測を行う
func (kmeans *MiniBatchKMeans) Predict(X mat.Matrix) (mat.Matrix, error) {
	kmeans.mu.RLock()
	defer kmeans.mu.RUnlock()

	if !kmeans.IsFitted() {
		return nil, errors.New("モデルが学習されていません")
	}

	rows, cols := X.Dims()
	if cols != kmeans.nFeatures_ {
		return nil, errors.Newf("特徴量の次元が一致しません: expected %d, got %d", kmeans.nFeatures_, cols)
	}

	predictions := mat.NewDense(rows, 1, nil)

	for i := 0; i < rows; i++ {
		sample := mat.Row(nil, i, X)
		cluster := kmeans.findNearestCluster(sample, kmeans.clusterCenters_)
		predictions.Set(i, 0, float64(cluster))
	}

	return predictions, nil
}

// FitPredict は学習と予測を同時に行う
func (kmeans *MiniBatchKMeans) FitPredict(X, y mat.Matrix) (mat.Matrix, error) {
	err := kmeans.Fit(X, y)
	if err != nil {
		return nil, err
	}
	return kmeans.Predict(X)
}

// ストリーミング学習メソッド

// FitStream はデータストリームからモデルを学習
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

// PredictStream は入力ストリームに対してリアルタイム予測
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

// インターフェース実装メソッド

// NIterations は実行された学習イテレーション数を返す
func (kmeans *MiniBatchKMeans) NIterations() int {
	kmeans.mu.RLock()
	defer kmeans.mu.RUnlock()
	return kmeans.nIter_
}

// IsWarmStart はウォームスタートが有効かどうかを返す（常にfalse）
func (kmeans *MiniBatchKMeans) IsWarmStart() bool {
	return false
}

// SetWarmStart はウォームスタートの有効/無効を設定（何もしない）
func (kmeans *MiniBatchKMeans) SetWarmStart(warmStart bool) {
	// MiniBatchKMeansはウォームスタートをサポートしない
}

// ClusterCenters は学習されたクラスタ中心を返す
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

// Labels は学習データのクラスタラベルを返す
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

// Inertia は慣性（クラスタ内平方和誤差）を返す
func (kmeans *MiniBatchKMeans) Inertia() float64 {
	kmeans.mu.RLock()
	defer kmeans.mu.RUnlock()
	return kmeans.inertia_
}

// 内部ヘルパーメソッド

// initializeCenters はクラスタ中心を初期化
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
		// デフォルトはk-means++
		return kmeans.initKMeansPlusPlus(X)
	}

	return centers
}

// initKMeansPlusPlus はk-means++初期化を実行
func (kmeans *MiniBatchKMeans) initKMeansPlusPlus(X mat.Matrix) [][]float64 {
	rows, cols := X.Dims()
	centers := make([][]float64, kmeans.nClusters)

	// 最初のクラスタ中心をランダムに選択
	centers[0] = make([]float64, cols)
	idx := kmeans.rng.Intn(rows)
	sample := mat.Row(nil, idx, X)
	copy(centers[0], sample)

	// 残りのクラスタ中心を選択
	for c := 1; c < kmeans.nClusters; c++ {
		distances := make([]float64, rows)
		totalDistance := 0.0

		// 各サンプルから最近傍クラスタ中心までの距離の二乗を計算
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

		// 確率に応じてサンプルを選択
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

// selectMiniBatch はミニバッチのサンプルインデックスを選択
func (kmeans *MiniBatchKMeans) selectMiniBatch(nSamples int) []int {
	batchSize := kmeans.batchSize
	if batchSize > nSamples {
		batchSize = nSamples
	}

	// ランダムサンプリング
	indices := make([]int, nSamples)
	for i := range indices {
		indices[i] = i
	}

	// Fisher-Yatesシャッフル
	for i := nSamples - 1; i > 0; i-- {
		j := kmeans.rng.Intn(i + 1)
		indices[i], indices[j] = indices[j], indices[i]
	}

	return indices[:batchSize]
}

// findNearestCluster は最近傍クラスタを検索
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

// computeInertia は慣性（クラスタ内平方和誤差）を計算
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

// 補助関数

// euclideanDistance はユークリッド距離を計算
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
