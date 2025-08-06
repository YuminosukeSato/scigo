package preprocessing

import (
	"fmt"
	"math"

	"github.com/YuminosukeSato/scigo/core/model"
	"github.com/YuminosukeSato/scigo/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

// StandardScaler はscikit-learn互換の標準化スケーラー
// データを平均0、標準偏差1に変換する
type StandardScaler struct {
	model.BaseEstimator
	
	// Mean は各特徴量の平均値
	Mean []float64
	
	// Scale は各特徴量の標準偏差
	Scale []float64
	
	// NFeatures は特徴量の数
	NFeatures int
	
	// WithMean は平均を引くかどうか (デフォルト: true)
	WithMean bool
	
	// WithStd は標準偏差で割るかどうか (デフォルト: true)
	WithStd bool
}

// NewStandardScaler は新しいStandardScalerを作成する
//
// パラメータ:
//   - withMean: 平均を引くかどうか (デフォルト: true)
//   - withStd: 標準偏差で割るかどうか (デフォルト: true)
//
// 戻り値:
//   - *StandardScaler: 新しいStandardScalerインスタンス
//
// 使用例:
//
//	scaler := preprocessing.NewStandardScaler(true, true)
//	err := scaler.Fit(X)
//	XScaled, err := scaler.Transform(X)
func NewStandardScaler(withMean, withStd bool) *StandardScaler {
	return &StandardScaler{
		WithMean: withMean,
		WithStd:  withStd,
	}
}

// NewStandardScalerDefault はデフォルト設定でStandardScalerを作成する
func NewStandardScalerDefault() *StandardScaler {
	return NewStandardScaler(true, true)
}

// Fit は訓練データから統計情報（平均、標準偏差）を計算する
//
// パラメータ:
//   - X: 訓練データ (n_samples × n_features の行列)
//
// 戻り値:
//   - error: エラーが発生した場合
func (s *StandardScaler) Fit(X mat.Matrix) error {
	r, c := X.Dims()
	if r == 0 || c == 0 {
		return errors.NewModelError("StandardScaler.Fit", "empty data", errors.ErrEmptyData)
	}
	
	s.NFeatures = c
	s.Mean = make([]float64, c)
	s.Scale = make([]float64, c)
	
	// 平均を計算
	if s.WithMean {
		for j := 0; j < c; j++ {
			sum := 0.0
			for i := 0; i < r; i++ {
				sum += X.At(i, j)
			}
			s.Mean[j] = sum / float64(r)
		}
	} else {
		// 平均を0に設定
		for j := 0; j < c; j++ {
			s.Mean[j] = 0.0
		}
	}
	
	// 標準偏差を計算
	if s.WithStd {
		for j := 0; j < c; j++ {
			sumSquares := 0.0
			for i := 0; i < r; i++ {
				diff := X.At(i, j) - s.Mean[j]
				sumSquares += diff * diff
			}
			variance := sumSquares / float64(r)
			s.Scale[j] = math.Sqrt(variance)
			
			// 標準偏差が0に近い場合は1に設定（ゼロ除算を避ける）
			if math.Abs(s.Scale[j]) < 1e-8 {
				s.Scale[j] = 1.0
			}
		}
	} else {
		// スケールを1に設定
		for j := 0; j < c; j++ {
			s.Scale[j] = 1.0
		}
	}
	
	s.SetFitted()
	return nil
}

// Transform は学習済みの統計情報を使ってデータを標準化する
//
// パラメータ:
//   - X: 変換するデータ
//
// 戻り値:
//   - mat.Matrix: 標準化されたデータ
//   - error: エラーが発生した場合
func (s *StandardScaler) Transform(X mat.Matrix) (mat.Matrix, error) {
	if !s.IsFitted() {
		return nil, errors.NewNotFittedError("StandardScaler", "Transform")
	}
	
	r, c := X.Dims()
	if c != s.NFeatures {
		return nil, errors.NewDimensionError("StandardScaler.Transform", s.NFeatures, c, 1)
	}
	
	// 結果を格納する行列を作成
	result := mat.NewDense(r, c, nil)
	
	// 各要素を標準化
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			value := X.At(i, j)
			standardized := (value - s.Mean[j]) / s.Scale[j]
			result.Set(i, j, standardized)
		}
	}
	
	return result, nil
}

// FitTransform は訓練データで学習し、同じデータを変換する
//
// パラメータ:
//   - X: 訓練・変換するデータ
//
// 戻り値:
//   - mat.Matrix: 標準化されたデータ
//   - error: エラーが発生した場合
func (s *StandardScaler) FitTransform(X mat.Matrix) (mat.Matrix, error) {
	if err := s.Fit(X); err != nil {
		return nil, err
	}
	return s.Transform(X)
}

// InverseTransform は標準化されたデータを元のスケールに戻す
//
// パラメータ:
//   - X: 標準化されたデータ
//
// 戻り値:
//   - mat.Matrix: 元のスケールに戻されたデータ
//   - error: エラーが発生した場合
func (s *StandardScaler) InverseTransform(X mat.Matrix) (mat.Matrix, error) {
	if !s.IsFitted() {
		return nil, errors.NewNotFittedError("StandardScaler", "InverseTransform")
	}
	
	r, c := X.Dims()
	if c != s.NFeatures {
		return nil, errors.NewDimensionError("StandardScaler.InverseTransform", s.NFeatures, c, 1)
	}
	
	// 結果を格納する行列を作成
	result := mat.NewDense(r, c, nil)
	
	// 各要素を逆変換
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			value := X.At(i, j)
			original := value*s.Scale[j] + s.Mean[j]
			result.Set(i, j, original)
		}
	}
	
	return result, nil
}

// GetParams はスケーラーのパラメータを取得する
func (s *StandardScaler) GetParams() map[string]interface{} {
	return map[string]interface{}{
		"with_mean": s.WithMean,
		"with_std":  s.WithStd,
	}
}

// String はスケーラーの文字列表現を返す
func (s *StandardScaler) String() string {
	if !s.IsFitted() {
		return fmt.Sprintf("StandardScaler(with_mean=%t, with_std=%t)", s.WithMean, s.WithStd)
	}
	return fmt.Sprintf("StandardScaler(with_mean=%t, with_std=%t, n_features=%d)", 
		s.WithMean, s.WithStd, s.NFeatures)
}

// MinMaxScaler はscikit-learn互換のMin-Maxスケーラー
// データを指定した範囲（デフォルト[0,1]）にスケーリングする
type MinMaxScaler struct {
	model.BaseEstimator
	
	// Min は各特徴量の最小値
	Min []float64
	
	// Max は各特徴量の最大値  
	Max []float64
	
	// Scale は各特徴量のスケール (max - min)
	Scale []float64
	
	// DataMin は学習データの最小値
	DataMin []float64
	
	// DataMax は学習データの最大値
	DataMax []float64
	
	// NFeatures は特徴量の数
	NFeatures int
	
	// FeatureRange はスケーリング後の範囲 [min, max]
	FeatureRange [2]float64
}

// NewMinMaxScaler は新しいMinMaxScalerを作成する
//
// パラメータ:
//   - featureRange: スケーリング後の範囲 [min, max] (デフォルト: [0, 1])
//
// 戻り値:
//   - *MinMaxScaler: 新しいMinMaxScalerインスタンス
//
// 使用例:
//
//	scaler := preprocessing.NewMinMaxScaler([2]float64{0.0, 1.0})
//	err := scaler.Fit(X)
//	XScaled, err := scaler.Transform(X)
func NewMinMaxScaler(featureRange [2]float64) *MinMaxScaler {
	return &MinMaxScaler{
		FeatureRange: featureRange,
	}
}

// NewMinMaxScalerDefault はデフォルト設定([0,1]範囲)でMinMaxScalerを作成する
func NewMinMaxScalerDefault() *MinMaxScaler {
	return NewMinMaxScaler([2]float64{0.0, 1.0})
}

// Fit は訓練データから最小値・最大値を計算する
//
// パラメータ:
//   - X: 訓練データ (n_samples × n_features の行列)
//
// 戻り値:
//   - error: エラーが発生した場合
func (m *MinMaxScaler) Fit(X mat.Matrix) error {
	r, c := X.Dims()
	if r == 0 || c == 0 {
		return errors.NewModelError("MinMaxScaler.Fit", "empty data", errors.ErrEmptyData)
	}
	
	m.NFeatures = c
	m.DataMin = make([]float64, c)
	m.DataMax = make([]float64, c)
	m.Min = make([]float64, c)
	m.Max = make([]float64, c)
	m.Scale = make([]float64, c)
	
	// 各特徴量の最小値・最大値を計算
	for j := 0; j < c; j++ {
		min := X.At(0, j)
		max := X.At(0, j)
		
		for i := 1; i < r; i++ {
			val := X.At(i, j)
			if val < min {
				min = val
			}
			if val > max {
				max = val
			}
		}
		
		m.DataMin[j] = min
		m.DataMax[j] = max
		
		// スケールを計算 (max - min)
		dataRange := max - min
		if math.Abs(dataRange) < 1e-8 {
			// 定数特徴量の場合、スケールを1に設定
			m.Scale[j] = 1.0
		} else {
			m.Scale[j] = dataRange
		}
		
		// 変換後の範囲を計算
		featureRange := m.FeatureRange[1] - m.FeatureRange[0]
		m.Min[j] = m.FeatureRange[0] - min * featureRange / m.Scale[j]
		m.Max[j] = m.FeatureRange[1] - max * featureRange / m.Scale[j]
	}
	
	m.SetFitted()
	return nil
}

// Transform は学習済みの統計情報を使ってデータをスケーリングする
//
// パラメータ:
//   - X: 変換するデータ
//
// 戻り値:
//   - mat.Matrix: スケーリングされたデータ
//   - error: エラーが発生した場合
func (m *MinMaxScaler) Transform(X mat.Matrix) (mat.Matrix, error) {
	if !m.IsFitted() {
		return nil, errors.NewNotFittedError("MinMaxScaler", "Transform")
	}
	
	r, c := X.Dims()
	if c != m.NFeatures {
		return nil, errors.NewDimensionError("MinMaxScaler.Transform", m.NFeatures, c, 1)
	}
	
	// 結果を格納する行列を作成
	result := mat.NewDense(r, c, nil)
	
	// 各要素をスケーリング
	featureRange := m.FeatureRange[1] - m.FeatureRange[0]
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			val := X.At(i, j)
			// X_scaled = X_std * (max - min) + min
			// where X_std = (X - X.min) / (X.max - X.min)
			scaled := (val - m.DataMin[j]) / m.Scale[j] * featureRange + m.FeatureRange[0]
			result.Set(i, j, scaled)
		}
	}
	
	return result, nil
}

// FitTransform は訓練データで学習し、同じデータを変換する
//
// パラメータ:
//   - X: 訓練・変換するデータ
//
// 戻り値:
//   - mat.Matrix: スケーリングされたデータ
//   - error: エラーが発生した場合
func (m *MinMaxScaler) FitTransform(X mat.Matrix) (mat.Matrix, error) {
	if err := m.Fit(X); err != nil {
		return nil, err
	}
	return m.Transform(X)
}

// InverseTransform はスケーリングされたデータを元の範囲に戻す
//
// パラメータ:
//   - X: スケーリングされたデータ
//
// 戻り値:
//   - mat.Matrix: 元の範囲に戻されたデータ
//   - error: エラーが発生した場合
func (m *MinMaxScaler) InverseTransform(X mat.Matrix) (mat.Matrix, error) {
	if !m.IsFitted() {
		return nil, errors.NewNotFittedError("MinMaxScaler", "InverseTransform")
	}
	
	r, c := X.Dims()
	if c != m.NFeatures {
		return nil, errors.NewDimensionError("MinMaxScaler.InverseTransform", m.NFeatures, c, 1)
	}
	
	// 結果を格納する行列を作成
	result := mat.NewDense(r, c, nil)
	
	// 各要素を逆変換
	featureRange := m.FeatureRange[1] - m.FeatureRange[0]
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			val := X.At(i, j)
			// 逆変換: X_orig = ((X_scaled - min) / (max - min)) * (data_max - data_min) + data_min
			original := ((val - m.FeatureRange[0]) / featureRange) * m.Scale[j] + m.DataMin[j]
			result.Set(i, j, original)
		}
	}
	
	return result, nil
}

// GetParams はスケーラーのパラメータを取得する
func (m *MinMaxScaler) GetParams() map[string]interface{} {
	return map[string]interface{}{
		"feature_range": m.FeatureRange,
	}
}

// String はスケーラーの文字列表現を返す
func (m *MinMaxScaler) String() string {
	if !m.IsFitted() {
		return fmt.Sprintf("MinMaxScaler(feature_range=[%.1f, %.1f])", 
			m.FeatureRange[0], m.FeatureRange[1])
	}
	return fmt.Sprintf("MinMaxScaler(feature_range=[%.1f, %.1f], n_features=%d)", 
		m.FeatureRange[0], m.FeatureRange[1], m.NFeatures)
}