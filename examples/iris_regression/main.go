package main

import (
	"encoding/csv"
	"fmt"
	"log/slog"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"

	"github.com/YuminosukeSato/scigo/linear"
)

// loadIrisData はCSVファイルからirisデータを読み込む
func loadIrisData(filename string) ([]float64, []float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer func() { _ = file.Close() }()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	var sepalLengths, sepalWidths []float64

	// ヘッダーをスキップして、データを読み込む
	for i := 1; i < len(records); i++ {
		sepalLength, err := strconv.ParseFloat(records[i][0], 64)
		if err != nil {
			continue
		}
		sepalWidth, err := strconv.ParseFloat(records[i][1], 64)
		if err != nil {
			continue
		}

		sepalLengths = append(sepalLengths, sepalLength)
		sepalWidths = append(sepalWidths, sepalWidth)
	}

	return sepalLengths, sepalWidths, nil
}

// createScatterPlot は散布図のプロッターを作成する
func createScatterPlot(xData, yData []float64) (*plotter.Scatter, error) {
	pts := make(plotter.XYs, len(xData))
	for i := range xData {
		pts[i].X = xData[i]
		pts[i].Y = yData[i]
	}

	scatter, err := plotter.NewScatter(pts)
	if err != nil {
		return nil, err
	}

	return scatter, nil
}

// createRegressionLine は回帰直線のプロッターを作成する
func createRegressionLine(xData []float64, lr *linear.LinearRegression) (*plotter.Line, error) {
	// 回帰直線の始点と終点を計算
	minX, maxX := xData[0], xData[0]
	for _, x := range xData {
		if x < minX {
			minX = x
		}
		if x > maxX {
			maxX = x
		}
	}

	// 回帰係数を取得
	weights := lr.Weights
	intercept := lr.Intercept

	// 直線の2点を計算
	pts := make(plotter.XYs, 2)
	pts[0].X = minX
	pts[0].Y = weights.At(0, 0)*minX + intercept
	pts[1].X = maxX
	pts[1].Y = weights.At(0, 0)*maxX + intercept

	line, err := plotter.NewLine(pts)
	if err != nil {
		return nil, err
	}

	return line, nil
}

func main() {
	// irisデータを読み込む
	sepalLengths, sepalWidths, err := loadIrisData("datasets/iris.csv")
	if err != nil {
		slog.Error("Failed to load iris data", "error", err)
		os.Exit(1)
	}

	fmt.Printf("Loaded %d data points\n", len(sepalLengths))

	// データを行列形式に変換
	n := len(sepalLengths)
	X := mat.NewDense(n, 1, nil)
	y := mat.NewDense(n, 1, nil)

	for i := 0; i < n; i++ {
		X.Set(i, 0, sepalLengths[i])
		y.Set(i, 0, sepalWidths[i])
	}

	// 線形回帰モデルを学習
	lr := linear.NewLinearRegression()
	err = lr.Fit(X, y)
	if err != nil {
		slog.Error("Failed to fit linear regression", "error", err)
		os.Exit(1)
	}

	// 回帰係数を表示
	fmt.Printf("Regression coefficient (slope): %.4f\n", lr.Weights.At(0, 0))
	fmt.Printf("Intercept: %.4f\n", lr.Intercept)

	// R²スコアを計算
	r2, err := lr.Score(X, y)
	if err != nil {
		slog.Error("Failed to calculate R² score", "error", err)
		os.Exit(1)
	}
	fmt.Printf("R² score: %.4f\n", r2)

	// プロットを作成
	p := plot.New()
	p.Title.Text = "Iris Dataset: Sepal Length vs Sepal Width"
	p.X.Label.Text = "Sepal Length (cm)"
	p.Y.Label.Text = "Sepal Width (cm)"

	// 散布図を追加
	scatter, err := createScatterPlot(sepalLengths, sepalWidths)
	if err != nil {
		slog.Error("Failed to create scatter plot", "error", err)
		os.Exit(1)
	}
	scatter.Color = plotter.DefaultLineStyle.Color
	p.Add(scatter)
	p.Legend.Add("Data points", scatter)

	// 回帰直線を追加
	line, err := createRegressionLine(sepalLengths, lr)
	if err != nil {
		slog.Error("Failed to create regression line", "error", err)
		os.Exit(1)
	}
	line.Width = vg.Points(2)
	line.Dashes = []vg.Length{}
	p.Add(line)
	p.Legend.Add("Regression line", line)

	// PNGファイルとして保存
	if err := p.Save(8*vg.Inch, 6*vg.Inch, "iris_regression.png"); err != nil {
		slog.Error("Failed to save plot", "error", err)
		os.Exit(1)
	}

	fmt.Println("Plot saved as iris_regression.png")
}
