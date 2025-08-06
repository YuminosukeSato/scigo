// Package lightgbm provides a pure Go implementation of LightGBM model inference
// with full compatibility for models trained in Python's LightGBM library.
//
// This package offers significant improvements over existing solutions like leaves:
//   - Full LightGBM compatibility: Reads all model formats (.txt, JSON, string)
//   - Training support: Not just inference, but also model training (future)
//   - scikit-learn compatible API: Familiar interface for Python users
//   - Convenience features: One-liner execution, auto-tuning, progress bars
//   - Numerical precision: Guaranteed float64 precision matching Python LightGBM
//
// # Basic Usage
//
// Load and use a pre-trained LightGBM model:
//
//	// Load model from file
//	model, err := lightgbm.LoadFromFile("model.txt")
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	// Make predictions
//	predictions := model.Predict(X)
//
// # Quick Training (One-liner)
//
// For maximum convenience, use the quick training functions:
//
//	// Train and predict in one line
//	result := lightgbm.QuickTrain(X, y).Predict(X_test)
//
//	// Automatic parameter tuning
//	model := lightgbm.AutoFit(X, y)
//
// # scikit-learn Compatible API
//
// Use familiar scikit-learn style classifiers and regressors:
//
//	// Classification
//	clf := lightgbm.NewLGBMClassifier()
//	clf.Fit(X_train, y_train)
//	predictions, _ := clf.Predict(X_test)
//	accuracy := clf.Score(X_test, y_test)
//
//	// Regression
//	reg := lightgbm.NewLGBMRegressor()
//	reg.Fit(X_train, y_train)
//	predictions, _ := reg.Predict(X_test)
//	r2 := reg.Score(X_test, y_test)
//
// # Model Loading Formats
//
// The package supports all LightGBM model formats:
//
//	// Text format (most common)
//	model, _ := lightgbm.LoadFromFile("model.txt")
//
//	// JSON format
//	model, _ := lightgbm.LoadFromJSON(jsonData)
//
//	// String format (model_to_string output)
//	model, _ := lightgbm.LoadFromString(modelString)
//
// # Pipeline Integration
//
// Seamlessly integrate with SciGo's pipeline system:
//
//	pipeline := pipeline.NewPipeline().
//	    Add("scaler", preprocessing.StandardScaler()).
//	    Add("model", lightgbm.NewLGBMClassifier()).
//	    Fit(X_train, y_train)
//
//	predictions := pipeline.Predict(X_test)
//
// # AutoML and Hyperparameter Tuning
//
// Automatic model optimization with time limits:
//
//	// Time-limited auto-tuning
//	model := lightgbm.AutoML(X, y).
//	    WithTimeLimit(5*time.Minute).
//	    WithCrossValidation(5).
//	    Optimize()
//
//	// Grid search with progress bar
//	best := lightgbm.GridSearchCV(paramGrid, X, y).
//	    WithProgressBar().
//	    Fit()
//
// # Progress Monitoring
//
// Track training progress in real-time:
//
//	model := lightgbm.NewLGBMClassifier().
//	    WithProgressBar().
//	    Fit(X, y)
//	// Output: [████████░░] 80% | Trees: 800/1000 | Loss: 0.234 | ETA: 2.3s
//
// # Visualization
//
// Built-in visualization capabilities:
//
//	// Feature importance
//	model.PlotImportance()       // ASCII chart
//	model.ExportImportanceJSON() // For D3.js visualization
//
//	// Tree visualization
//	model.PlotTree(0)           // First tree
//	model.ExportTreeGraphviz(0) // Graphviz format
//
// # Numerical Precision
//
// This package guarantees numerical precision matching Python's LightGBM:
//   - All computations use float64
//   - Deterministic mode available for reproducible results
//   - Careful handling of floating-point operations
//
// # Compatibility
//
// The package maintains compatibility with:
//   - Python LightGBM models (all versions)
//   - leaves library API (drop-in replacement)
//   - scikit-learn API conventions
//   - SciGo ecosystem (pipelines, preprocessing, etc.)
//
// # Performance
//
// Optimized for production use:
//   - Efficient tree traversal algorithms
//   - Parallel prediction for batch processing
//   - Memory-efficient model storage
//   - Optional GPU acceleration (future)
//
// For more examples and detailed documentation, see the individual type
// and function documentation below.
package lightgbm