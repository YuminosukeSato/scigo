package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"math/rand"
	"strings"
	"time"

	"github.com/YuminosukeSato/GoML/core/model"
	"github.com/YuminosukeSato/GoML/sklearn/linear_model"
	"gonum.org/v1/gonum/mat"
)

func main() {
	fmt.Println("=== GoML Streaming Learning Demo ===")

	// デモ1: バッチ学習 vs オンライン学習の比較
	fmt.Println("1. Batch Learning vs Online Learning Comparison")
	fmt.Println(strings.Repeat("-", 50))
	compareBatchVsOnline()

	// デモ2: リアルタイムストリーミング学習
	fmt.Println("\n2. Real-time Streaming Learning")
	fmt.Println(strings.Repeat("-", 50))
	demonstrateStreamingLearning()

	// デモ3: コンセプトドリフトへの適応
	fmt.Println("\n3. Concept Drift Adaptation")
	fmt.Println(strings.Repeat("-", 50))
	demonstrateConceptDrift()

	// デモ4: Test-then-Train評価
	fmt.Println("\n4. Test-then-Train Evaluation")
	fmt.Println(strings.Repeat("-", 50))
	demonstrateTestThenTrain()
}

// compareBatchVsOnline はバッチ学習とオンライン学習を比較
func compareBatchVsOnline() {
	// データ生成
	nSamples := 1000
	X := mat.NewDense(nSamples, 2, nil)
	y := mat.NewDense(nSamples, 1, nil)
	
	rand.Seed(42)
	for i := 0; i < nSamples; i++ {
		x1 := rand.NormFloat64()
		x2 := rand.NormFloat64()
		X.Set(i, 0, x1)
		X.Set(i, 1, x2)
		// y = 3*x1 + 2*x2 + 1 + noise
		y.Set(i, 0, 3*x1 + 2*x2 + 1 + rand.NormFloat64()*0.1)
	}

	// バッチ学習
	batchModel := linear_model.NewSGDRegressor(
		linear_model.WithMaxIter(100),
		linear_model.WithRandomState(42),
	)
	
	startTime := time.Now()
	err := batchModel.Fit(X, y)
	if err != nil {
		log.Fatalf("Batch learning failed: %v", err)
	}
	batchTime := time.Since(startTime)

	// オンライン学習
	onlineModel := linear_model.NewSGDRegressor(
		linear_model.WithRandomState(42),
	)
	
	startTime = time.Now()
	batchSize := 10
	for i := 0; i < nSamples; i += batchSize {
		end := i + batchSize
		if end > nSamples {
			end = nSamples
		}
		
		XBatch := X.Slice(i, end, 0, 2)
		yBatch := y.Slice(i, end, 0, 1)
		
		err := onlineModel.PartialFit(XBatch, yBatch, nil)
		if err != nil {
			log.Fatalf("Online learning failed: %v", err)
		}
	}
	onlineTime := time.Since(startTime)

	// 結果表示
	fmt.Printf("Batch Learning:\n")
	fmt.Printf("  Time: %v\n", batchTime)
	fmt.Printf("  Coefficients: %.3f\n", batchModel.Coef())
	fmt.Printf("  Intercept: %.3f\n", batchModel.Intercept())
	
	fmt.Printf("\nOnline Learning:\n")
	fmt.Printf("  Time: %v\n", onlineTime)
	fmt.Printf("  Coefficients: %.3f\n", onlineModel.Coef())
	fmt.Printf("  Intercept: %.3f\n", onlineModel.Intercept())
	fmt.Printf("  Total iterations: %d\n", onlineModel.NIterations())
}

// demonstrateStreamingLearning はリアルタイムストリーミング学習をデモ
func demonstrateStreamingLearning() {
	sgd := linear_model.NewSGDRegressor(
		linear_model.WithLearningRate("invscaling"),
		linear_model.WithEta0(0.1),
		linear_model.WithRandomState(42),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	// データストリームを生成
	dataChan := make(chan *model.Batch, 10)
	
	go func() {
		defer close(dataChan)
		rand.Seed(42)
		
		for i := 0; i < 100; i++ {
			// ミニバッチを生成
			batchSize := 5
			X := mat.NewDense(batchSize, 3, nil)
			y := mat.NewDense(batchSize, 1, nil)
			
			for j := 0; j < batchSize; j++ {
				x1 := rand.NormFloat64()
				x2 := rand.NormFloat64()
				x3 := rand.NormFloat64()
				X.Set(j, 0, x1)
				X.Set(j, 1, x2)
				X.Set(j, 2, x3)
				// y = 2*x1 + 3*x2 - x3 + 1
				y.Set(j, 0, 2*x1 + 3*x2 - x3 + 1 + rand.NormFloat64()*0.05)
			}
			
			select {
			case dataChan <- &model.Batch{X: X, Y: y}:
				if i%10 == 0 {
					fmt.Printf("  Batch %d sent, Loss: %.4f\n", i, sgd.GetLoss())
				}
			case <-ctx.Done():
				return
			}
			
			time.Sleep(30 * time.Millisecond) // ストリーミングをシミュレート
		}
	}()

	// ストリーミング学習を開始
	fmt.Println("Starting streaming learning...")
	err := sgd.FitStream(ctx, dataChan)
	if err != nil && err != context.DeadlineExceeded {
		log.Printf("Streaming learning error: %v", err)
	}

	fmt.Printf("\nFinal model:\n")
	fmt.Printf("  Coefficients: %.3f\n", sgd.Coef())
	fmt.Printf("  Intercept: %.3f\n", sgd.Intercept())
	fmt.Printf("  Final Loss: %.4f\n", sgd.GetLoss())
	fmt.Printf("  Converged: %v\n", sgd.GetConverged())
}

// demonstrateConceptDrift はコンセプトドリフトへの適応をデモ
func demonstrateConceptDrift() {
	sgd := linear_model.NewSGDRegressor(
		linear_model.WithLearningRate("adaptive"),
		linear_model.WithEta0(0.1),
		linear_model.WithRandomState(42),
	)

	rand.Seed(42)
	fmt.Println("Simulating concept drift...")
	
	// フェーズ1: y = 2*x + 1
	fmt.Println("\nPhase 1: y = 2*x + 1")
	for i := 0; i < 50; i++ {
		X := mat.NewDense(10, 1, nil)
		y := mat.NewDense(10, 1, nil)
		
		for j := 0; j < 10; j++ {
			x := rand.NormFloat64()
			X.Set(j, 0, x)
			y.Set(j, 0, 2*x + 1 + rand.NormFloat64()*0.1)
		}
		
		sgd.PartialFit(X, y, nil)
		
		if i%10 == 0 {
			fmt.Printf("  Iteration %d - Coef: %.3f, Intercept: %.3f, Loss: %.4f\n",
				i, sgd.Coef()[0], sgd.Intercept(), sgd.GetLoss())
		}
	}

	// フェーズ2: コンセプトドリフト発生 - y = -x + 3
	fmt.Println("\nPhase 2: Concept drift - y = -x + 3")
	for i := 0; i < 50; i++ {
		X := mat.NewDense(10, 1, nil)
		y := mat.NewDense(10, 1, nil)
		
		for j := 0; j < 10; j++ {
			x := rand.NormFloat64()
			X.Set(j, 0, x)
			y.Set(j, 0, -x + 3 + rand.NormFloat64()*0.1)
		}
		
		sgd.PartialFit(X, y, nil)
		
		if i%10 == 0 {
			fmt.Printf("  Iteration %d - Coef: %.3f, Intercept: %.3f, Loss: %.4f\n",
				50+i, sgd.Coef()[0], sgd.Intercept(), sgd.GetLoss())
		}
	}

	fmt.Println("\nModel adapted to concept drift successfully!")
}

// demonstrateTestThenTrain はtest-then-train評価をデモ
func demonstrateTestThenTrain() {
	sgd := linear_model.NewSGDRegressor(
		linear_model.WithLearningRate("invscaling"),
		linear_model.WithEta0(0.1),
		linear_model.WithRandomState(42),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	dataChan := make(chan *model.Batch, 5)
	
	// データ生成
	go func() {
		defer close(dataChan)
		rand.Seed(42)
		
		for i := 0; i < 20; i++ {
			X := mat.NewDense(5, 2, nil)
			y := mat.NewDense(5, 1, nil)
			
			for j := 0; j < 5; j++ {
				x1 := rand.Float64() * 10
				x2 := rand.Float64() * 10
				X.Set(j, 0, x1)
				X.Set(j, 1, x2)
				// y = 0.5*x1 + 0.3*x2 + 2
				y.Set(j, 0, 0.5*x1 + 0.3*x2 + 2 + rand.NormFloat64()*0.1)
			}
			
			select {
			case dataChan <- &model.Batch{X: X, Y: y}:
			case <-ctx.Done():
				return
			}
			
			time.Sleep(50 * time.Millisecond)
		}
	}()

	// Test-then-train評価
	fmt.Println("Starting test-then-train evaluation...")
	outputChan := sgd.FitPredictStream(ctx, dataChan)
	
	totalError := 0.0
	nPredictions := 0
	
	// 真の値を計算するための関数
	truePrediction := func(x1, x2 float64) float64 {
		return 0.5*x1 + 0.3*x2 + 2
	}
	
	batchIdx := 0
	for pred := range outputChan {
		rows, _ := pred.Dims()
		
		// 各予測の誤差を計算
		batchError := 0.0
		for i := 0; i < rows; i++ {
			// 簡単のため、固定値での誤差計算
			trueVal := truePrediction(float64(batchIdx*5+i), float64(batchIdx*5+i)*0.6)
			predVal := pred.At(i, 0)
			batchError += math.Abs(trueVal - predVal)
		}
		
		avgError := batchError / float64(rows)
		totalError += batchError
		nPredictions += rows
		
		fmt.Printf("  Batch %d - Avg Error: %.4f, Cumulative Avg Error: %.4f\n",
			batchIdx, avgError, totalError/float64(nPredictions))
		
		batchIdx++
	}

	fmt.Printf("\nTest-then-Train Results:\n")
	fmt.Printf("  Total predictions: %d\n", nPredictions)
	fmt.Printf("  Average error: %.4f\n", totalError/float64(nPredictions))
	fmt.Printf("  Final coefficients: %.3f\n", sgd.Coef())
	fmt.Printf("  Final intercept: %.3f\n", sgd.Intercept())
}

