package linear

// Option is a function that configures LinearRegression
type Option func(*SKLinearRegression)

// WithFitIntercept sets whether to calculate the intercept
func WithFitIntercept(fit bool) Option {
	return func(lr *SKLinearRegression) {
		lr.fitIntercept = fit
	}
}

// WithCopyX sets whether to copy X matrix
func WithCopyX(copy bool) Option {
	return func(lr *SKLinearRegression) {
		lr.copyX = copy
	}
}

// WithTol sets the tolerance for the optimization
func WithTol(tol float64) Option {
	return func(lr *SKLinearRegression) {
		lr.tol = tol
	}
}

// WithNJobs sets the number of parallel jobs
func WithNJobs(n int) Option {
	return func(lr *SKLinearRegression) {
		lr.nJobs = n
	}
}

// WithPositive constrains the coefficients to be positive
func WithPositive(positive bool) Option {
	return func(lr *SKLinearRegression) {
		lr.positive = positive
	}
}