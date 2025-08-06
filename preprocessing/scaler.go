// Package preprocessing provides data preprocessing utilities for machine learning.
//
// This package implements scikit-learn compatible preprocessing components including:
//
//   - StandardScaler: Standardizes features by removing the mean and scaling to unit variance
//   - MinMaxScaler: Transforms features by scaling each feature to a given range
//   - OneHotEncoder: Encodes categorical features as one-hot numeric arrays
//
// All preprocessing components follow the scikit-learn API pattern with Fit, Transform, 
// and FitTransform methods. They integrate seamlessly with the BaseEstimator pattern 
// for consistent state management and serialization support.
//
// Example usage:
//
//	scaler := preprocessing.NewStandardScaler(true, true)
//	err := scaler.Fit(trainingData)
//	if err != nil {
//		log.Fatal(err)
//	}
//	scaledData, err := scaler.Transform(testData)
//
// The package is designed for production machine learning pipelines with emphasis
// on memory efficiency, thread safety, and compatibility with popular ML libraries.
package preprocessing

import (
	"fmt"
	"math"

	"github.com/YuminosukeSato/scigo/core/model"
	scigoErrors "github.com/YuminosukeSato/scigo/pkg/errors"
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

// NewStandardScaler creates a new StandardScaler for feature standardization.
//
// StandardScaler transforms features by removing the mean and scaling to unit variance.
// This is a common preprocessing step that ensures all features contribute equally
// to machine learning algorithms and improves numerical stability.
//
// Parameters:
//   - withMean: whether to center the data at zero by removing the mean (default: true)
//   - withStd: whether to scale the data to unit variance by dividing by standard deviation (default: true)
//
// Returns:
//   - *StandardScaler: A new StandardScaler instance ready for fitting
//
// Example:
//   // Standard z-score normalization (mean=0, std=1)
//   scaler := preprocessing.NewStandardScaler(true, true)
//   err := scaler.Fit(X_train)
//   X_scaled, err := scaler.Transform(X_test)
//
//   // Scale only (keep original mean)
//   scaler := preprocessing.NewStandardScaler(false, true)
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

// Fit computes the statistics (mean and scale) from the training data.
//
// This method calculates the feature-wise mean and standard deviation from the
// provided training data, which will be used for future transformations.
// The scaler must be fitted before calling Transform or InverseTransform.
//
// Parameters:
//   - X: Training data matrix of shape (n_samples, n_features)
//
// Returns:
//   - error: nil if successful, otherwise an error describing the failure
//
// Errors:
//   - ErrEmptyData: if X is empty
//   - ErrDimensionMismatch: if X has inconsistent dimensions
//
// Example:
//   scaler := preprocessing.NewStandardScaler(true, true)
//   err := scaler.Fit(trainingData)
//   if err != nil {
//       log.Fatal(err)
//   }
func (s *StandardScaler) Fit(X mat.Matrix) (err error) {
	defer scigoErrors.Recover(&err, "StandardScaler.Fit")
	r, c := X.Dims()
	if r == 0 || c == 0 {
		return scigoErrors.NewModelError("StandardScaler.Fit", "empty data", scigoErrors.ErrEmptyData)
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

// Transform applies standardization to the input data using fitted statistics.
//
// This method standardizes features by removing the mean and scaling to unit
// variance using the statistics computed during the Fit phase. The transformation
// formula is: X_scaled = (X - mean) / scale.
//
// Parameters:
//   - X: Input data matrix of shape (n_samples, n_features)
//
// Returns:
//   - mat.Matrix: Standardized data matrix with same shape as input
//   - error: nil if successful, otherwise an error describing the failure
//
// Errors:
//   - ErrNotFitted: if the scaler hasn't been fitted yet
//   - ErrDimensionMismatch: if X doesn't match the number of features from training
//
// Example:
//   scaledData, err := scaler.Transform(testData)
//   if err != nil {
//       log.Fatal(err)
//   }
func (s *StandardScaler) Transform(X mat.Matrix) (_ mat.Matrix, err error) {
	defer scigoErrors.Recover(&err, "StandardScaler.Transform")
	if !s.IsFitted() {
		return nil, scigoErrors.NewNotFittedError("StandardScaler", "Transform")
	}
	
	r, c := X.Dims()
	if c != s.NFeatures {
		return nil, scigoErrors.NewDimensionError("StandardScaler.Transform", s.NFeatures, c, 1)
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

// FitTransform fits the scaler and transforms the training data in one step.
//
// This convenience method combines Fit and Transform operations, computing
// statistics from the input data and immediately applying the transformation.
// Equivalent to calling Fit(X) followed by Transform(X).
//
// Parameters:
//   - X: Training data matrix of shape (n_samples, n_features)
//
// Returns:
//   - mat.Matrix: Standardized training data matrix
//   - error: nil if successful, otherwise an error from either fitting or transformation
//
// Example:
//   scaler := preprocessing.NewStandardScaler(true, true)
//   scaledTraining, err := scaler.FitTransform(trainingData)
//   if err != nil {
//       log.Fatal(err)
//   }
//   // Now scaler is fitted and can transform new data
//   scaledTest, err := scaler.Transform(testData)
func (s *StandardScaler) FitTransform(X mat.Matrix) (_ mat.Matrix, err error) {
	defer scigoErrors.Recover(&err, "StandardScaler.FitTransform")
	if err := s.Fit(X); err != nil {
		return nil, err
	}
	return s.Transform(X)
}

// InverseTransform reverses the standardization transformation.
//
// This method transforms standardized data back to the original scale using
// the fitted statistics. The inverse transformation formula is:
// X_orig = X_scaled * scale + mean.
//
// Parameters:
//   - X: Standardized data matrix of shape (n_samples, n_features)
//
// Returns:
//   - mat.Matrix: Data matrix in original scale
//   - error: nil if successful, otherwise an error describing the failure
//
// Errors:
//   - ErrNotFitted: if the scaler hasn't been fitted yet
//   - ErrDimensionMismatch: if X doesn't match the number of features from training
//
// Example:
//   originalData, err := scaler.InverseTransform(scaledData)
//   if err != nil {
//       log.Fatal(err)
//   }
func (s *StandardScaler) InverseTransform(X mat.Matrix) (_ mat.Matrix, err error) {
	defer scigoErrors.Recover(&err, "StandardScaler.InverseTransform")
	if !s.IsFitted() {
		return nil, scigoErrors.NewNotFittedError("StandardScaler", "InverseTransform")
	}
	
	r, c := X.Dims()
	if c != s.NFeatures {
		return nil, scigoErrors.NewDimensionError("StandardScaler.InverseTransform", s.NFeatures, c, 1)
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

// NewMinMaxScaler creates a new MinMaxScaler for feature scaling.
//
// MinMaxScaler transforms features by scaling each feature to a given range.
// The transformation is given by: X_scaled = (X - X.min) / (X.max - X.min) * (max - min) + min
//
// Parameters:
//   - featureRange: Target range for scaling [min, max] (typically [0, 1] or [-1, 1])
//
// Returns:
//   - *MinMaxScaler: A new MinMaxScaler instance ready for fitting
//
// Example:
//   // Scale to [0, 1] range
//   scaler := preprocessing.NewMinMaxScaler([2]float64{0.0, 1.0})
//   err := scaler.Fit(trainingData)
//   scaledData, err := scaler.Transform(testData)
//   
//   // Scale to [-1, 1] range
//   scaler := preprocessing.NewMinMaxScaler([2]float64{-1.0, 1.0})
func NewMinMaxScaler(featureRange [2]float64) *MinMaxScaler {
	return &MinMaxScaler{
		FeatureRange: featureRange,
	}
}

// NewMinMaxScalerDefault はデフォルト設定([0,1]範囲)でMinMaxScalerを作成する
func NewMinMaxScalerDefault() *MinMaxScaler {
	return NewMinMaxScaler([2]float64{0.0, 1.0})
}

// Fit computes the minimum and maximum values for each feature from training data.
//
// This method calculates the feature-wise minimum and maximum values that will be
// used for scaling transformations. The scaler must be fitted before calling
// Transform or InverseTransform.
//
// Parameters:
//   - X: Training data matrix of shape (n_samples, n_features)
//
// Returns:
//   - error: nil if successful, otherwise an error describing the failure
//
// Errors:
//   - ErrEmptyData: if X is empty
//
// Example:
//   scaler := preprocessing.NewMinMaxScaler([2]float64{0.0, 1.0})
//   err := scaler.Fit(trainingData)
//   if err != nil {
//       log.Fatal(err)
//   }
func (m *MinMaxScaler) Fit(X mat.Matrix) (err error) {
	defer scigoErrors.Recover(&err, "MinMaxScaler.Fit")
	r, c := X.Dims()
	if r == 0 || c == 0 {
		return scigoErrors.NewModelError("MinMaxScaler.Fit", "empty data", scigoErrors.ErrEmptyData)
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

// Transform scales input data to the fitted feature range.
//
// This method transforms data using the minimum and maximum values computed during
// the Fit phase. Each feature is independently scaled to the target range.
//
// Parameters:
//   - X: Input data matrix of shape (n_samples, n_features)
//
// Returns:
//   - mat.Matrix: Scaled data matrix with values in the target range
//   - error: nil if successful, otherwise an error describing the failure
//
// Errors:
//   - ErrNotFitted: if the scaler hasn't been fitted yet
//   - ErrDimensionMismatch: if X doesn't match the number of features from training
//
// Example:
//   scaledData, err := scaler.Transform(testData)
//   if err != nil {
//       log.Fatal(err)
//   }
//   // scaledData values are now in the range specified during NewMinMaxScaler
func (m *MinMaxScaler) Transform(X mat.Matrix) (_ mat.Matrix, err error) {
	defer scigoErrors.Recover(&err, "MinMaxScaler.Transform")
	if !m.IsFitted() {
		return nil, scigoErrors.NewNotFittedError("MinMaxScaler", "Transform")
	}
	
	r, c := X.Dims()
	if c != m.NFeatures {
		return nil, scigoErrors.NewDimensionError("MinMaxScaler.Transform", m.NFeatures, c, 1)
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

// FitTransform fits the scaler and transforms the training data in one step.
//
// This convenience method combines Fit and Transform operations, computing
// min/max statistics from the input data and immediately applying the scaling.
// Equivalent to calling Fit(X) followed by Transform(X).
//
// Parameters:
//   - X: Training data matrix of shape (n_samples, n_features)
//
// Returns:
//   - mat.Matrix: Scaled training data matrix in the target range
//   - error: nil if successful, otherwise an error from either fitting or transformation
//
// Example:
//   scaler := preprocessing.NewMinMaxScaler([2]float64{0.0, 1.0})
//   scaledTraining, err := scaler.FitTransform(trainingData)
//   if err != nil {
//       log.Fatal(err)
//   }
//   // Now scaler is fitted and can transform new data
//   scaledTest, err := scaler.Transform(testData)
func (m *MinMaxScaler) FitTransform(X mat.Matrix) (_ mat.Matrix, err error) {
	defer scigoErrors.Recover(&err, "MinMaxScaler.FitTransform")
	if err := m.Fit(X); err != nil {
		return nil, err
	}
	return m.Transform(X)
}

// InverseTransform reverses the min-max scaling transformation.
//
// This method transforms scaled data back to the original range using the
// fitted min/max statistics. Useful for interpreting results or recovering
// original data values.
//
// Parameters:
//   - X: Scaled data matrix of shape (n_samples, n_features)
//
// Returns:
//   - mat.Matrix: Data matrix in original scale and range
//   - error: nil if successful, otherwise an error describing the failure
//
// Errors:
//   - ErrNotFitted: if the scaler hasn't been fitted yet
//   - ErrDimensionMismatch: if X doesn't match the number of features from training
//
// Example:
//   originalData, err := scaler.InverseTransform(scaledData)
//   if err != nil {
//       log.Fatal(err)
//   }
func (m *MinMaxScaler) InverseTransform(X mat.Matrix) (_ mat.Matrix, err error) {
	defer scigoErrors.Recover(&err, "MinMaxScaler.InverseTransform")
	if !m.IsFitted() {
		return nil, scigoErrors.NewNotFittedError("MinMaxScaler", "InverseTransform")
	}
	
	r, c := X.Dims()
	if c != m.NFeatures {
		return nil, scigoErrors.NewDimensionError("MinMaxScaler.InverseTransform", m.NFeatures, c, 1)
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