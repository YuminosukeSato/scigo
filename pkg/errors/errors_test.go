package errors

import (
	"fmt"
	"strings"
	"testing"
)

func TestNewModelError(t *testing.T) {
	tests := []struct {
		name     string
		op       string
		kind     string
		err      error
		wantMsg  string
		hasStack bool
	}{
		{
			name:     "with original error",
			op:       "Fit",
			kind:     "invalid input",
			err:      fmt.Errorf("test error"),
			wantMsg:  "goml: Fit: invalid input: test error",
			hasStack: true,
		},
		{
			name:     "without original error",
			op:       "Predict",
			kind:     "not fitted",
			err:      nil,
			wantMsg:  "goml: Predict: not fitted",
			hasStack: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := NewModelError(tt.op, tt.kind, tt.err)

			// 基本的なエラーメッセージの確認
			if err.Error() != tt.wantMsg {
				t.Errorf("Error() = %v, want %v", err.Error(), tt.wantMsg)
			}

			// スタックトレースの存在確認
			if tt.hasStack {
				formatted := fmt.Sprintf("%+v", err)
				if !strings.Contains(formatted, "errors_test.go") {
					t.Error("Expected stack trace to contain test file name")
				}
			}

			// ModelError型にキャスト可能か確認
			var modelErr *ModelError
			if !As(err, &modelErr) {
				t.Error("Error should be castable to *ModelError")
			}

			// ModelError型へのキャストのみ確認
		})
	}
}

func TestNewDimensionError(t *testing.T) {
	err := NewDimensionError("Predict", 10, 10, 0)

	// 基本的なエラーメッセージの確認
	want := "goml: Predict: dimension mismatch on axis 0 (rows). Expected 10, got 10"
	if err.Error() != want {
		t.Errorf("Error() = %v, want %v", err.Error(), want)
	}

	// DimensionError型にキャスト可能か確認
	var dimErr *DimensionError
	if !As(err, &dimErr) {
		t.Error("Error should be castable to *DimensionError")
	}

	// DimensionError型へのキャストのみ確認
}

func TestNewNotFittedError(t *testing.T) {
	err := NewNotFittedError("LinearRegression", "Predict")

	// 基本的なエラーメッセージの確認
	want := "goml: LinearRegression: this model is not fitted yet. Call Fit() before using Predict()"
	if err.Error() != want {
		t.Errorf("Error() = %v, want %v", err.Error(), want)
	}

	// NotFittedError型にキャスト可能か確認
	var notFittedErr *NotFittedError
	if !As(err, &notFittedErr) {
		t.Error("Error should be castable to *NotFittedError")
	}
}

func TestNewValueError(t *testing.T) {
	tests := []struct {
		name    string
		op      string
		param   string
		value   interface{}
		message string
		wantMsg string
	}{
		{
			name:    "with message",
			op:      "SetParam",
			param:   "learning_rate",
			value:   -0.5,
			message: "must be positive",
			wantMsg: "goml: SetParam: learning_rate: -0.5 (must be positive)",
		},
		{
			name:    "without message",
			op:      "SetParam",
			param:   "n_components",
			value:   0,
			message: "",
			wantMsg: "goml: SetParam: n_components: 0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var err error
			if tt.message != "" {
				err = NewValueError(tt.op, fmt.Sprintf("%s: %v (%s)", tt.param, tt.value, tt.message))
			} else {
				err = NewValueError(tt.op, fmt.Sprintf("%s: %v", tt.param, tt.value))
			}

			if err.Error() != tt.wantMsg {
				t.Errorf("Error() = %v, want %v", err.Error(), tt.wantMsg)
			}

			// ValueError型にキャスト可能か確認
			var valErr *ValueError
			if !As(err, &valErr) {
				t.Error("Error should be castable to *ValueError")
			}
		})
	}
}

func TestNewConvergenceWarning(t *testing.T) {
	warn := NewConvergenceWarning("GradientDescent", 1000, "loss did not decrease")

	// 基本的なエラーメッセージの確認
	want := "GradientDescent failed to converge after 1000 iterations: loss did not decrease"
	if warn.Error() != want {
		t.Errorf("Error() = %v, want %v", warn.Error(), want)
	}

	// ConvergenceWarning型へのキャストのみ確認
	var convWarn *ConvergenceWarning
	if !As(warn, &convWarn) {
		t.Error("Warning should be castable to *ConvergenceWarning")
	}
}

func TestWrapAndIs(t *testing.T) {
	// 元のエラー
	baseErr := ErrNotImplemented

	// ラップ
	wrapped := Wrap(baseErr, "in LinearRegression.Predict")

	// Is関数でチェック
	if !Is(wrapped, ErrNotImplemented) {
		t.Error("Expected Is(wrapped, ErrNotImplemented) to be true")
	}

	// エラーメッセージの確認
	if !strings.Contains(wrapped.Error(), "in LinearRegression.Predict") {
		t.Error("Expected wrapped error to contain wrapping message")
	}
}

func TestWrapf(t *testing.T) {
	// 元のエラー
	baseErr := ErrEmptyData

	// フォーマット付きラップ
	wrapped := Wrapf(baseErr, "in %s: expected %d, got %d", "Predict", 10, 5)

	// Is関数でチェック
	if !Is(wrapped, ErrEmptyData) {
		t.Error("Expected Is(wrapped, ErrEmptyData) to be true")
	}

	// エラーメッセージの確認
	expectedMsg := "in Predict: expected 10, got 5"
	if !strings.Contains(wrapped.Error(), expectedMsg) {
		t.Errorf("Expected wrapped error to contain %q", expectedMsg)
	}
}

func TestErrorChaining(t *testing.T) {
	// エラーチェーンの作成
	err1 := fmt.Errorf("base error")
	err2 := Wrap(err1, "wrapped once")
	err3 := NewModelError("Operation", "failed", err2)

	// チェーン全体を確認
	if !strings.Contains(err3.Error(), "base error") {
		t.Error("Expected error chain to contain base error")
	}

	// スタックトレースの確認（詳細表示）
	formatted := fmt.Sprintf("%+v", err3)
	if !strings.Contains(formatted, "errors_test.go") {
		t.Error("Expected detailed error to contain stack trace")
	}
}

// GetStackTraceは削除されたためテストも削除

// GetDetailsは削除されたためテストも削除

// GetAllDetailsは削除されたためテストも削除
