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
			
			// デバッグ情報の確認
			details := GetAllDetails(err)
			if len(details) == 0 {
				t.Error("Expected error to have details")
			}
		})
	}
}

func TestNewDimensionError(t *testing.T) {
	err := NewDimensionError("Predict", []int{10, 5}, []int{10, 3})
	
	// 基本的なエラーメッセージの確認
	want := "goml: Predict: dimension mismatch: expected [10 5], got [10 3]"
	if err.Error() != want {
		t.Errorf("Error() = %v, want %v", err.Error(), want)
	}
	
	// DimensionError型にキャスト可能か確認
	var dimErr *DimensionError
	if !As(err, &dimErr) {
		t.Error("Error should be castable to *DimensionError")
	}
	
	// デバッグ情報の確認
	details := GetAllDetails(err)
	if details == nil {
		t.Error("Expected error to have details")
	}
}

func TestNewNotFittedError(t *testing.T) {
	err := NewNotFittedError("LinearRegression")
	
	// 基本的なエラーメッセージの確認
	want := "goml: LinearRegression is not fitted yet. Call Fit before Predict"
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
			wantMsg: "goml: SetParam: invalid value for learning_rate: -0.5 (must be positive)",
		},
		{
			name:    "without message",
			op:      "SetParam",
			param:   "n_components",
			value:   0,
			message: "",
			wantMsg: "goml: SetParam: invalid value for n_components: 0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := NewValueError(tt.op, tt.param, tt.value, tt.message)
			
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

func TestNewConvergenceError(t *testing.T) {
	err := NewConvergenceError("GradientDescent", 1000, "loss did not decrease")
	
	// 基本的なエラーメッセージの確認
	want := "goml: GradientDescent failed to converge after 1000 iterations: loss did not decrease"
	if err.Error() != want {
		t.Errorf("Error() = %v, want %v", err.Error(), want)
	}
	
	// ConvergenceError型にキャスト可能か確認
	var convErr *ConvergenceError
	if !As(err, &convErr) {
		t.Error("Error should be castable to *ConvergenceError")
	}
}

func TestWrapAndIs(t *testing.T) {
	// 元のエラー
	baseErr := ErrNotFitted
	
	// ラップ
	wrapped := Wrap(baseErr, "in LinearRegression.Predict")
	
	// Is関数でチェック
	if !Is(wrapped, ErrNotFitted) {
		t.Error("Expected Is(wrapped, ErrNotFitted) to be true")
	}
	
	// エラーメッセージの確認
	if !strings.Contains(wrapped.Error(), "in LinearRegression.Predict") {
		t.Error("Expected wrapped error to contain wrapping message")
	}
}

func TestWrapf(t *testing.T) {
	// 元のエラー
	baseErr := ErrDimensionMismatch
	
	// フォーマット付きラップ
	wrapped := Wrapf(baseErr, "in %s: expected %d, got %d", "Predict", 10, 5)
	
	// Is関数でチェック
	if !Is(wrapped, ErrDimensionMismatch) {
		t.Error("Expected Is(wrapped, ErrDimensionMismatch) to be true")
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

func TestGetStackTrace(t *testing.T) {
	err := NewModelError("TestOp", "test kind", nil)
	
	stackTrace := GetStackTrace(err)
	if stackTrace == "" {
		t.Error("Expected GetStackTrace to return non-empty stack trace")
	}
}

func TestGetDetails(t *testing.T) {
	err := NewDimensionError("TestOp", []int{5, 3}, []int{5, 2})
	
	detail := GetDetails(err)
	if detail == nil {
		t.Fatal("Expected GetDetails to return non-nil ErrorDetail")
	}
	
	if detail.Message == "" {
		t.Error("Expected ErrorDetail to have non-empty Message")
	}
	
	if detail.StackTrace == "" {
		t.Error("Expected ErrorDetail to have non-empty StackTrace")
	}
}

func TestGetAllDetails(t *testing.T) {
	err := NewValueError("TestOp", "param", 42, "test message")
	
	details := GetAllDetails(err)
	if details == nil {
		t.Fatal("Expected GetAllDetails to return non-nil map")
	}
	
	if _, ok := details["message"]; !ok {
		t.Error("Expected details map to contain 'message' key")
	}
	
	if _, ok := details["stacktrace"]; !ok {
		t.Error("Expected details map to contain 'stacktrace' key")
	}
}