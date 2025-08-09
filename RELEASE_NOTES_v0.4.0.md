## 🎉 SciGo v0.4.0 Release

### ✨ New Features

#### 📊 LogisticRegression
- Binary and multiclass classification (one-vs-rest)
- Gradient descent optimization with L2 regularization
- Probability predictions with PredictProba
- Full scikit-learn API compatibility

#### 🌲 DecisionTreeClassifier  
- CART algorithm with Gini and Entropy criteria
- Feature importance calculation
- Max depth and min samples constraints
- Multiclass classification support
- Tree structure introspection (GetDepth, GetNLeaves)

### 🔧 Improvements

#### CI/CD Enhancements
- Automatic go fmt checking in CI
- Local CI execution capability for faster development
- Enhanced security scanning with semgrep
- Improved linter configuration

#### Documentation
- Complete English translation of all code comments
- Comprehensive English documentation
- Enhanced API documentation

### 🔄 Changes
- Refactored codebase to use composition over inheritance pattern
- Improved error handling and error message capitalization per Go conventions

### 🐛 Bug Fixes
- Fixed test stability for XOR pattern in DecisionTree
- Resolved convergence issues in LogisticRegression tests
- Corrected error message capitalization to follow Go conventions
- Fixed various linter warnings and issues

### 📦 Installation
```bash
go get github.com/YuminosukeSato/scigo@v0.4.0
```

### 📚 Documentation
Full documentation available at: https://pkg.go.dev/github.com/YuminosukeSato/scigo

### 🚀 What's Next
- v0.5.0: RandomForestClassifier, RandomForestRegressor, and SVM implementation
- v0.6.0: XGBoost integration with Python model compatibility
- v0.7.0: LightGBM native training implementation

Thank you to all contributors! 🙏