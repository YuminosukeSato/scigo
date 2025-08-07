package drift

import (
	"math"
	"sync"
)

// DDM (Drift Detection Method) is a concept drift detection method
// Proposed in J. Gama, P. Medas, G. Castillo, P. Rodrigues (2004)
// "Learning with Drift Detection"
type DDM struct {
	// Hyperparameters
	minNumInstances int     // Minimum number of instances
	warningLevel    float64 // Warning level
	outControlLevel float64 // Out of control level

	// Statistics
	numInstances int     // Number of instances
	numErrors    int     // Number of errors
	errorRate    float64 // Error rate
	stdDev       float64 // Standard deviation

	// Reference values (minimum values from learning start)
	minErrorRate float64 // Minimum error rate
	minStdDev    float64 // Minimum standard deviation

	// State
	warningDetected bool // Warning detection flag
	driftDetected   bool // Drift detection flag

	// Internal state
	mu sync.RWMutex
}

// DriftDetectionResult represents the result of drift detection
type DriftDetectionResult struct {
	WarningDetected bool    // Whether warning was detected
	DriftDetected   bool    // Whether drift was detected
	ErrorRate       float64 // Current error rate
	ConfidenceLevel float64 // Confidence level
}

// NewDDM creates a new DDM instance
func NewDDM(options ...DDMOption) *DDM {
	ddm := &DDM{
		minNumInstances: 30,
		warningLevel:    2.0, // μ + 2σ
		outControlLevel: 3.0, // μ + 3σ
		minErrorRate:    math.Inf(1),
		minStdDev:       math.Inf(1),
	}

	for _, opt := range options {
		opt(ddm)
	}

	return ddm
}

// DDMOption is a DDM configuration option
type DDMOption func(*DDM)

// WithDDMMinNumInstances sets the minimum number of samples
func WithDDMMinNumInstances(n int) DDMOption {
	return func(ddm *DDM) {
		ddm.minNumInstances = n
	}
}

// WithDDMWarningLevel sets the warning level
func WithDDMWarningLevel(level float64) DDMOption {
	return func(ddm *DDM) {
		ddm.warningLevel = level
	}
}

// WithDDMOutControlLevel sets the out-of-control level
func WithDDMOutControlLevel(level float64) DDMOption {
	return func(ddm *DDM) {
		ddm.outControlLevel = level
	}
}

// Update updates the drift detector with prediction results
// correct: whether the prediction was correct
// return: drift detection result
func (ddm *DDM) Update(correct bool) *DriftDetectionResult {
	ddm.mu.Lock()
	defer ddm.mu.Unlock()

	ddm.numInstances++
	if !correct {
		ddm.numErrors++
	}

	// Do not detect if minimum sample size is not reached
	if ddm.numInstances < ddm.minNumInstances {
		return &DriftDetectionResult{
			WarningDetected: false,
			DriftDetected:   false,
			ErrorRate:       0,
			ConfidenceLevel: 0,
		}
	}

	// Calculate error rate and standard deviation
	ddm.errorRate = float64(ddm.numErrors) / float64(ddm.numInstances)
	ddm.stdDev = math.Sqrt(ddm.errorRate * (1.0 - ddm.errorRate) / float64(ddm.numInstances))

	result := &DriftDetectionResult{
		ErrorRate: ddm.errorRate,
	}

	// 基準値の更新（最小エラー率とその時の標準偏差）
	currentLevel := ddm.errorRate + ddm.stdDev
	if currentLevel < (ddm.minErrorRate + ddm.minStdDev) {
		ddm.minErrorRate = ddm.errorRate
		ddm.minStdDev = ddm.stdDev
	}

	// 信頼度の計算
	if ddm.minStdDev > 0 {
		result.ConfidenceLevel = (ddm.errorRate + ddm.stdDev) / (ddm.minErrorRate + ddm.minStdDev)
	} else {
		result.ConfidenceLevel = 1.0
	}

	// 警告レベルの検出
	warningThreshold := ddm.minErrorRate + ddm.warningLevel*ddm.minStdDev
	if ddm.errorRate+ddm.stdDev > warningThreshold {
		ddm.warningDetected = true
		result.WarningDetected = true
	} else {
		ddm.warningDetected = false
	}

	// ドリフトレベルの検出
	driftThreshold := ddm.minErrorRate + ddm.outControlLevel*ddm.minStdDev
	if ddm.errorRate+ddm.stdDev > driftThreshold {
		ddm.driftDetected = true
		result.DriftDetected = true
		// ドリフト検出時はリセット
		ddm.resetAfterDrift()
	} else {
		ddm.driftDetected = false
	}

	return result
}

// UpdateWithPrediction は予測値と実際値でドリフト検出器を更新
func (ddm *DDM) UpdateWithPrediction(predicted, actual float64) *DriftDetectionResult {
	correct := math.Abs(predicted-actual) < 1e-10 // 分類問題の場合
	return ddm.Update(correct)
}

// UpdateWithError はエラー値でドリフト検出器を更新
func (ddm *DDM) UpdateWithError(error float64) *DriftDetectionResult {
	// Consider as correct if error is below threshold
	correct := error < 0.1
	return ddm.Update(correct)
}

// Reset はドリフト検出器をリセット
func (ddm *DDM) Reset() {
	ddm.mu.Lock()
	defer ddm.mu.Unlock()

	ddm.numInstances = 0
	ddm.numErrors = 0
	ddm.errorRate = 0
	ddm.stdDev = 0
	ddm.minErrorRate = math.Inf(1)
	ddm.minStdDev = math.Inf(1)
	ddm.warningDetected = false
	ddm.driftDetected = false
}

// resetAfterDrift はドリフト検出後のリセット
func (ddm *DDM) resetAfterDrift() {
	ddm.numInstances = 0
	ddm.numErrors = 0
	ddm.errorRate = 0
	ddm.stdDev = 0
	ddm.minErrorRate = math.Inf(1)
	ddm.minStdDev = math.Inf(1)
	ddm.warningDetected = false
	ddm.driftDetected = false
}

// GetStatistics は現在の統計情報を返す
func (ddm *DDM) GetStatistics() DDMStatistics {
	ddm.mu.RLock()
	defer ddm.mu.RUnlock()

	return DDMStatistics{
		NumInstances:    ddm.numInstances,
		NumErrors:       ddm.numErrors,
		ErrorRate:       ddm.errorRate,
		StdDev:          ddm.stdDev,
		MinErrorRate:    ddm.minErrorRate,
		MinStdDev:       ddm.minStdDev,
		WarningDetected: ddm.warningDetected,
		DriftDetected:   ddm.driftDetected,
	}
}

// DDMStatistics はDDMの統計情報
type DDMStatistics struct {
	NumInstances    int     // サンプル数
	NumErrors       int     // エラー数
	ErrorRate       float64 // エラー率
	StdDev          float64 // 標準偏差
	MinErrorRate    float64 // 最小エラー率
	MinStdDev       float64 // 最小標準偏差
	WarningDetected bool    // 警告検出フラグ
	DriftDetected   bool    // ドリフト検出フラグ
}

// ADWIN (Adaptive Windowing) はアダプティブウィンドウによるドリフト検出
// A. Bifet, R. Gavalda (2007) "Learning from time-changing data with adaptive windowing"
type ADWIN struct {
	// ハイパーパラメータ
	delta      float64 // 信頼度パラメータ（小さいほど敏感）
	maxBuckets int     // 最大バケット数

	// データ構造
	buckets    []bucket // バケット
	totalSum   float64  // 全体の合計
	totalCount int      // 全体のサンプル数
	width      int      // ウィンドウ幅
	variance   float64  // 分散

	// 内部状態
	mu sync.RWMutex
}

// bucket はADWINで使用するバケット
type bucket struct {
	sum   float64 // バケット内の合計
	count int     // バケット内のサンプル数
}

// NewADWIN は新しいADWINを作成
func NewADWIN(options ...ADWINOption) *ADWIN {
	adwin := &ADWIN{
		delta:      0.002, // デフォルトの信頼度
		maxBuckets: 1000,
		buckets:    make([]bucket, 0),
	}

	for _, opt := range options {
		opt(adwin)
	}

	return adwin
}

// ADWINOption はADWINの設定オプション
type ADWINOption func(*ADWIN)

// WithADWINDelta は信頼度パラメータを設定
func WithADWINDelta(delta float64) ADWINOption {
	return func(adwin *ADWIN) {
		adwin.delta = delta
	}
}

// WithADWINMaxBuckets は最大バケット数を設定
func WithADWINMaxBuckets(max int) ADWINOption {
	return func(adwin *ADWIN) {
		adwin.maxBuckets = max
	}
}

// Update は新しい値でADWINを更新
func (adwin *ADWIN) Update(value float64) bool {
	adwin.mu.Lock()
	defer adwin.mu.Unlock()

	// 新しい値を追加
	adwin.addElement(value)

	// ドリフト検出
	return adwin.detectDrift()
}

// addElement は新しい要素を追加
func (adwin *ADWIN) addElement(value float64) {
	// 新しいバケットを作成
	newBucket := bucket{sum: value, count: 1}

	// 既存のバケットとマージする可能性をチェック
	if len(adwin.buckets) > 0 {
		lastBucket := &adwin.buckets[len(adwin.buckets)-1]
		// 同じサイズのバケットがある場合はマージ
		if lastBucket.count == newBucket.count {
			lastBucket.sum += newBucket.sum
			lastBucket.count += newBucket.count
		} else {
			adwin.buckets = append(adwin.buckets, newBucket)
		}
	} else {
		adwin.buckets = append(adwin.buckets, newBucket)
	}

	// 全体の統計を更新
	adwin.totalSum += value
	adwin.totalCount++
	adwin.width++

	// バケット数の制限
	if len(adwin.buckets) > adwin.maxBuckets {
		adwin.removeOldestBucket()
	}
}

// detectDrift はドリフトを検出
func (adwin *ADWIN) detectDrift() bool {
	if len(adwin.buckets) < 2 {
		return false
	}

	n := adwin.totalCount
	if n < 5 {
		return false
	}

	// 異なる分割点でドリフトをチェック
	for i := 1; i < len(adwin.buckets); i++ {
		n0, n1, mean0, mean1 := adwin.getSubwindowStats(i)

		if n0 <= 0 || n1 <= 0 {
			continue
		}

		// ホフディングの不等式に基づく検定
		diff := math.Abs(mean0 - mean1)
		bound := adwin.getHoeffdingBound(n0, n1)

		if diff > bound {
			// ドリフト検出：古いバケットを削除
			adwin.buckets = adwin.buckets[i:]
			adwin.updateGlobalStats()
			return true
		}
	}

	return false
}

// getSubwindowStats は分割点での統計を取得
func (adwin *ADWIN) getSubwindowStats(splitPoint int) (int, int, float64, float64) {
	// 左側（古い）ウィンドウ
	sum0, count0 := 0.0, 0
	for i := 0; i < splitPoint; i++ {
		sum0 += adwin.buckets[i].sum
		count0 += adwin.buckets[i].count
	}

	// 右側（新しい）ウィンドウ
	sum1, count1 := 0.0, 0
	for i := splitPoint; i < len(adwin.buckets); i++ {
		sum1 += adwin.buckets[i].sum
		count1 += adwin.buckets[i].count
	}

	mean0 := 0.0
	if count0 > 0 {
		mean0 = sum0 / float64(count0)
	}

	mean1 := 0.0
	if count1 > 0 {
		mean1 = sum1 / float64(count1)
	}

	return count0, count1, mean0, mean1
}

// getHoeffdingBound はホフディング境界を計算
func (adwin *ADWIN) getHoeffdingBound(n0, n1 int) float64 {
	if n0 <= 0 || n1 <= 0 {
		return 0
	}

	m := 1.0/float64(n0) + 1.0/float64(n1)
	return math.Sqrt(0.5 * m * math.Log(2.0/adwin.delta))
}

// removeOldestBucket は最も古いバケットを削除
func (adwin *ADWIN) removeOldestBucket() {
	if len(adwin.buckets) == 0 {
		return
	}

	oldest := adwin.buckets[0]
	adwin.totalSum -= oldest.sum
	adwin.totalCount -= oldest.count
	adwin.width -= oldest.count

	adwin.buckets = adwin.buckets[1:]
}

// updateGlobalStats は全体の統計を更新
func (adwin *ADWIN) updateGlobalStats() {
	adwin.totalSum = 0
	adwin.totalCount = 0
	adwin.width = 0

	for _, bucket := range adwin.buckets {
		adwin.totalSum += bucket.sum
		adwin.totalCount += bucket.count
		adwin.width += bucket.count
	}
}

// GetMean は現在のウィンドウの平均を返す
func (adwin *ADWIN) GetMean() float64 {
	adwin.mu.RLock()
	defer adwin.mu.RUnlock()

	if adwin.totalCount == 0 {
		return 0
	}
	return adwin.totalSum / float64(adwin.totalCount)
}

// GetWidth は現在のウィンドウ幅を返す
func (adwin *ADWIN) GetWidth() int {
	adwin.mu.RLock()
	defer adwin.mu.RUnlock()
	return adwin.width
}

// Reset はADWINをリセット
func (adwin *ADWIN) Reset() {
	adwin.mu.Lock()
	defer adwin.mu.Unlock()

	adwin.buckets = make([]bucket, 0)
	adwin.totalSum = 0
	adwin.totalCount = 0
	adwin.width = 0
	adwin.variance = 0
}

// DriftDetector はドリフト検出器の共通インターフェース
type DriftDetector interface {
	// Update は新しい値でドリフト検出器を更新
	Update(value float64) bool

	// Reset はドリフト検出器をリセット
	Reset()
}
