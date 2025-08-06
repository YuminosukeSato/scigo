package tensor

import (
	"github.com/YuminosukeSato/scigo/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

// Tensor はgonum/mat.Denseをラップした多次元配列構造体
type Tensor struct {
	data  *mat.Dense
	shape []int
}

// NewTensor は新しいテンソルを作成する
func NewTensor(data []float64, shape ...int) (*Tensor, error) {
	if len(shape) == 0 {
		return nil, errors.NewValueError("NewTensor", "shape must be provided")
	}

	size := 1
	for _, s := range shape {
		if s <= 0 {
			return nil, errors.NewValueError("NewTensor", "all dimensions must be positive")
		}
		size *= s
	}

	if len(data) != size {
		return nil, errors.NewDimensionError("NewTensor", size, len(data), 0)
	}

	// 現在は2次元のみサポート
	if len(shape) != 2 {
		return nil, errors.NewValueError("NewTensor", "currently only 2D tensors are supported")
	}

	return &Tensor{
		data:  mat.NewDense(shape[0], shape[1], data),
		shape: shape,
	}, nil
}

// NewTensorFromDense はmat.Denseから新しいテンソルを作成する
func NewTensorFromDense(dense *mat.Dense) *Tensor {
	r, c := dense.Dims()
	return &Tensor{
		data:  dense,
		shape: []int{r, c},
	}
}

// NewZeros は指定された形状のゼロテンソルを作成する
func NewZeros(shape ...int) (*Tensor, error) {
	if len(shape) != 2 {
		return nil, errors.NewValueError("NewZeros", "currently only 2D tensors are supported")
	}

	size := shape[0] * shape[1]
	data := make([]float64, size)
	return NewTensor(data, shape...)
}

// NewOnes は指定された形状の1で埋められたテンソルを作成する
func NewOnes(shape ...int) (*Tensor, error) {
	if len(shape) != 2 {
		return nil, errors.NewValueError("NewOnes", "currently only 2D tensors are supported")
	}

	size := shape[0] * shape[1]
	data := make([]float64, size)
	for i := range data {
		data[i] = 1.0
	}
	return NewTensor(data, shape...)
}

// Shape はテンソルの形状を返す
func (t *Tensor) Shape() []int {
	return append([]int{}, t.shape...)
}

// Dims はテンソルの次元（行数、列数）を返す
func (t *Tensor) Dims() (int, int) {
	return t.data.Dims()
}

// At は指定された位置の値を返す
func (t *Tensor) At(i, j int) float64 {
	return t.data.At(i, j)
}

// Set は指定された位置に値を設定する
func (t *Tensor) Set(i, j int, v float64) {
	t.data.Set(i, j, v)
}

// Data は内部のmat.Denseへの参照を返す
// 注意: 直接操作は推奨されない
func (t *Tensor) Data() *mat.Dense {
	return t.data
}

// RawData は内部データのコピーをスライスとして返す
func (t *Tensor) RawData() []float64 {
	r, c := t.data.Dims()
	data := make([]float64, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			data[i*c+j] = t.data.At(i, j)
		}
	}
	return data
}

// Copy はテンソルのディープコピーを作成する
func (t *Tensor) Copy() *Tensor {
	var newData mat.Dense
	newData.CloneFrom(t.data)
	return &Tensor{
		data:  &newData,
		shape: append([]int{}, t.shape...),
	}
}

// T はテンソルの転置を返す
func (t *Tensor) T() *Tensor {
	return &Tensor{
		data:  t.data.T().(*mat.Dense),
		shape: []int{t.shape[1], t.shape[0]},
	}
}

// Reshape はテンソルの形状を変更する
func (t *Tensor) Reshape(shape ...int) error {
	if len(shape) != 2 {
		return errors.Newf("currently only 2D reshaping is supported")
	}

	newSize := shape[0] * shape[1]
	oldSize := t.shape[0] * t.shape[1]

	if newSize != oldSize {
		return errors.Newf("cannot reshape tensor of size %d to size %d", oldSize, newSize)
	}

	// データをコピーして新しい形状で再構築
	rawData := t.RawData()
	newDense := mat.NewDense(shape[0], shape[1], rawData)

	t.data = newDense
	t.shape = shape

	return nil
}

// Slice は指定された範囲のスライスを返す
func (t *Tensor) Slice(rowStart, rowEnd, colStart, colEnd int) (*Tensor, error) {
	r, c := t.data.Dims()

	if rowStart < 0 || rowEnd > r || colStart < 0 || colEnd > c {
		return nil, errors.Newf("slice indices out of bounds")
	}

	if rowStart >= rowEnd || colStart >= colEnd {
		return nil, errors.Newf("invalid slice range")
	}

	sliced := t.data.Slice(rowStart, rowEnd, colStart, colEnd).(*mat.Dense)
	return NewTensorFromDense(sliced), nil
}
