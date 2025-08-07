# Clustering Guide

Comprehensive guide to clustering algorithms in SciGo for unsupervised learning tasks.

## Overview

Clustering is an unsupervised learning technique that groups similar data points together. SciGo provides efficient implementations of popular clustering algorithms optimized for production use.

## Clustering Algorithms

| Algorithm | Type | Use Case | Pros | Cons |
|-----------|------|----------|------|------|
| K-Means | Centroid-based | General clustering | Fast, scalable | Assumes spherical clusters |
| MiniBatch K-Means | Centroid-based | Large datasets | Very fast, online learning | Less accurate than K-Means |
| DBSCAN | Density-based | Arbitrary shapes | Finds outliers, no k needed | Sensitive to parameters |
| Hierarchical | Connectivity-based | Dendrograms | No k needed, interpretable | O(n²) complexity |
| Gaussian Mixture | Model-based | Soft clustering | Probabilistic, flexible shapes | Can overfit |

## K-Means Clustering

### Basic Usage

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/YuminosukeSato/scigo/sklearn/cluster"
    "github.com/YuminosukeSato/scigo/metrics"
    "gonum.org/v1/gonum/mat"
)

func main() {
    // Load data
    X := loadData() // Shape: (n_samples, n_features)
    
    // Create K-Means clusterer
    kmeans := cluster.NewKMeans(
        5,  // Number of clusters
        cluster.WithMaxIter(300),
        cluster.WithInit("k-means++"),
        cluster.WithRandomState(42),
    )
    
    // Fit the model
    err := kmeans.Fit(X)
    if err != nil {
        log.Fatal("Clustering failed:", err)
    }
    
    // Get cluster assignments
    labels := kmeans.Labels()
    
    // Get cluster centers
    centers := kmeans.ClusterCenters()
    
    // Calculate inertia (sum of squared distances)
    inertia := kmeans.Inertia()
    fmt.Printf("Inertia: %.2f\n", inertia)
    
    // Predict cluster for new points
    XNew := mat.NewDense(10, X.(*mat.Dense).RawMatrix().Cols, nil)
    newLabels, _ := kmeans.Predict(XNew)
    
    // Transform to cluster-distance space
    distances, _ := kmeans.Transform(XNew)
}
```

### K-Means++ Initialization

```go
// K-Means++ selects initial centers intelligently
func kMeansPlusPlus(X mat.Matrix, k int) []int {
    rows, _ := X.Dims()
    centers := make([]int, k)
    
    // Choose first center randomly
    centers[0] = rand.Intn(rows)
    
    for i := 1; i < k; i++ {
        // Calculate distances to nearest center
        distances := make([]float64, rows)
        for j := 0; j < rows; j++ {
            minDist := math.Inf(1)
            for c := 0; c < i; c++ {
                dist := euclideanDistance(
                    X.(*mat.Dense).RowView(j),
                    X.(*mat.Dense).RowView(centers[c]),
                )
                if dist < minDist {
                    minDist = dist
                }
            }
            distances[j] = minDist * minDist
        }
        
        // Choose next center with probability proportional to distance²
        centers[i] = weightedChoice(distances)
    }
    
    return centers
}
```

### Elbow Method for Optimal K

```go
func findOptimalK(X mat.Matrix, maxK int) int {
    inertias := make([]float64, maxK-1)
    
    for k := 2; k <= maxK; k++ {
        kmeans := cluster.NewKMeans(k)
        kmeans.Fit(X)
        inertias[k-2] = kmeans.Inertia()
    }
    
    // Find elbow point
    elbowK := findElbowPoint(inertias) + 2
    
    fmt.Printf("Optimal K: %d\n", elbowK)
    
    // Plot inertias
    for k := 2; k <= maxK; k++ {
        fmt.Printf("K=%d: Inertia=%.2f\n", k, inertias[k-2])
    }
    
    return elbowK
}

func findElbowPoint(values []float64) int {
    // Calculate second derivative
    n := len(values)
    if n < 3 {
        return 0
    }
    
    maxCurvature := 0.0
    elbowIdx := 0
    
    for i := 1; i < n-1; i++ {
        curvature := values[i-1] - 2*values[i] + values[i+1]
        if curvature > maxCurvature {
            maxCurvature = curvature
            elbowIdx = i
        }
    }
    
    return elbowIdx
}
```

## MiniBatch K-Means

### Streaming Clustering

```go
func streamingClustering() {
    // Create MiniBatch K-Means for large datasets
    mbkmeans := cluster.NewMiniBatchKMeans(
        10,  // Number of clusters
        cluster.WithBatchSize(256),
        cluster.WithMaxIter(100),
    )
    
    // Process data in batches
    for batch := range dataStream {
        err := mbkmeans.PartialFit(batch)
        if err != nil {
            log.Printf("Batch update failed: %v", err)
            continue
        }
        
        // Monitor convergence
        if mbkmeans.NIterations() % 10 == 0 {
            fmt.Printf("Iteration %d: Inertia=%.2f\n",
                mbkmeans.NIterations(),
                mbkmeans.Inertia())
        }
    }
    
    // Final clusters
    centers := mbkmeans.ClusterCenters()
    fmt.Printf("Final clusters: %d\n", centers.(*mat.Dense).RawMatrix().Rows)
}
```

### Online Learning

```go
type OnlineKMeans struct {
    centers  *mat.Dense
    counts   []int
    k        int
}

func (km *OnlineKMeans) Update(x mat.Vector) {
    // Find nearest center
    nearest := km.findNearestCenter(x)
    
    // Update center using running average
    km.counts[nearest]++
    alpha := 1.0 / float64(km.counts[nearest])
    
    center := km.centers.RowView(nearest).(*mat.VecDense)
    for i := 0; i < center.Len(); i++ {
        oldVal := center.AtVec(i)
        newVal := oldVal + alpha*(x.AtVec(i)-oldVal)
        center.SetVec(i, newVal)
    }
}
```

## DBSCAN

### Density-Based Clustering

```go
type DBSCAN struct {
    eps      float64  // Maximum distance between points
    minPts   int      // Minimum points to form dense region
    metric   string   // Distance metric
}

func NewDBSCAN(eps float64, minPts int) *DBSCAN {
    return &DBSCAN{
        eps:    eps,
        minPts: minPts,
        metric: "euclidean",
    }
}

func (db *DBSCAN) Fit(X mat.Matrix) ([]int, error) {
    rows, _ := X.Dims()
    labels := make([]int, rows)
    for i := range labels {
        labels[i] = -1  // Unassigned
    }
    
    clusterID := 0
    
    for i := 0; i < rows; i++ {
        if labels[i] != -1 {
            continue  // Already processed
        }
        
        // Find neighbors
        neighbors := db.findNeighbors(X, i)
        
        if len(neighbors) < db.minPts {
            labels[i] = -2  // Noise point
            continue
        }
        
        // Start new cluster
        labels[i] = clusterID
        db.expandCluster(X, labels, i, neighbors, clusterID)
        clusterID++
    }
    
    return labels, nil
}

func (db *DBSCAN) findNeighbors(X mat.Matrix, pointIdx int) []int {
    neighbors := []int{}
    point := X.(*mat.Dense).RowView(pointIdx)
    rows, _ := X.Dims()
    
    for i := 0; i < rows; i++ {
        if i == pointIdx {
            continue
        }
        
        dist := euclideanDistance(point, X.(*mat.Dense).RowView(i))
        if dist <= db.eps {
            neighbors = append(neighbors, i)
        }
    }
    
    return neighbors
}

func (db *DBSCAN) expandCluster(X mat.Matrix, labels []int, pointIdx int,
    neighbors []int, clusterID int) {
    
    seeds := append([]int{}, neighbors...)
    
    for len(seeds) > 0 {
        currentPoint := seeds[0]
        seeds = seeds[1:]
        
        if labels[currentPoint] == -2 {  // Was noise
            labels[currentPoint] = clusterID
            continue
        }
        
        if labels[currentPoint] != -1 {  // Already in cluster
            continue
        }
        
        labels[currentPoint] = clusterID
        
        // Find neighbors of current point
        currentNeighbors := db.findNeighbors(X, currentPoint)
        
        if len(currentNeighbors) >= db.minPts {
            seeds = append(seeds, currentNeighbors...)
        }
    }
}
```

## Hierarchical Clustering

### Agglomerative Clustering

```go
type AgglomerativeClustering struct {
    nClusters int
    linkage   string  // "single", "complete", "average"
    affinity  string  // "euclidean", "cosine"
}

func (ac *AgglomerativeClustering) Fit(X mat.Matrix) ([]int, *Dendrogram) {
    rows, _ := X.Dims()
    
    // Initialize each point as its own cluster
    clusters := make([][]int, rows)
    for i := range clusters {
        clusters[i] = []int{i}
    }
    
    // Build dendrogram
    dendrogram := &Dendrogram{}
    
    // Compute initial distance matrix
    distances := ac.computeDistanceMatrix(X)
    
    // Merge clusters until target number reached
    for len(clusters) > ac.nClusters {
        // Find closest clusters
        i, j := ac.findClosestClusters(distances)
        
        // Merge clusters
        clusters[i] = append(clusters[i], clusters[j]...)
        clusters = append(clusters[:j], clusters[j+1:]...)
        
        // Update distances
        ac.updateDistances(distances, i, j)
        
        // Record merge in dendrogram
        dendrogram.AddMerge(i, j, distances[i][j])
    }
    
    // Assign labels
    labels := make([]int, rows)
    for clusterID, cluster := range clusters {
        for _, pointIdx := range cluster {
            labels[pointIdx] = clusterID
        }
    }
    
    return labels, dendrogram
}

func (ac *AgglomerativeClustering) computeDistanceMatrix(X mat.Matrix) [][]float64 {
    rows, _ := X.Dims()
    distances := make([][]float64, rows)
    
    for i := range distances {
        distances[i] = make([]float64, rows)
        for j := range distances[i] {
            if i != j {
                distances[i][j] = ac.distance(
                    X.(*mat.Dense).RowView(i),
                    X.(*mat.Dense).RowView(j),
                )
            }
        }
    }
    
    return distances
}
```

### Dendrogram Visualization

```go
type Dendrogram struct {
    merges []Merge
}

type Merge struct {
    cluster1 int
    cluster2 int
    distance float64
    size     int
}

func (d *Dendrogram) CutTree(height float64) []int {
    // Cut dendrogram at specified height
    clusters := d.getClustersAtHeight(height)
    return clusters
}

func (d *Dendrogram) Print() {
    fmt.Println("Dendrogram:")
    for _, merge := range d.merges {
        fmt.Printf("Merge clusters %d and %d at distance %.3f\n",
            merge.cluster1, merge.cluster2, merge.distance)
    }
}
```

## Gaussian Mixture Models

### Soft Clustering

```go
type GaussianMixture struct {
    nComponents int
    means       []*mat.VecDense
    covariances []*mat.Dense
    weights     []float64
}

func (gm *GaussianMixture) Fit(X mat.Matrix) error {
    // Initialize using K-Means
    kmeans := cluster.NewKMeans(gm.nComponents)
    kmeans.Fit(X)
    
    // Initialize parameters
    gm.initializeFromKMeans(X, kmeans)
    
    // EM algorithm
    for iter := 0; iter < 100; iter++ {
        // E-step: Calculate responsibilities
        responsibilities := gm.eStep(X)
        
        // M-step: Update parameters
        gm.mStep(X, responsibilities)
        
        // Check convergence
        if gm.hasConverged() {
            break
        }
    }
    
    return nil
}

func (gm *GaussianMixture) PredictProba(X mat.Matrix) mat.Matrix {
    rows, _ := X.Dims()
    proba := mat.NewDense(rows, gm.nComponents, nil)
    
    for i := 0; i < rows; i++ {
        x := X.(*mat.Dense).RowView(i)
        
        for j := 0; j < gm.nComponents; j++ {
            // Calculate probability for component j
            p := gm.weights[j] * gm.gaussianPDF(x, gm.means[j], gm.covariances[j])
            proba.Set(i, j, p)
        }
        
        // Normalize
        rowSum := mat.Sum(proba.RowView(i))
        for j := 0; j < gm.nComponents; j++ {
            proba.Set(i, j, proba.At(i, j)/rowSum)
        }
    }
    
    return proba
}
```

## Clustering Evaluation

### Internal Metrics

```go
// Silhouette Score (-1 to 1, higher is better)
func evaluateSilhouette(X mat.Matrix, labels []int) float64 {
    score, _ := metrics.SilhouetteScore(X, labels)
    return score
}

// Davies-Bouldin Index (lower is better)
func evaluateDaviesBouldin(X mat.Matrix, labels []int) float64 {
    score, _ := metrics.DaviesBouldinScore(X, labels)
    return score
}

// Calinski-Harabasz Index (higher is better)
func evaluateCalinskiHarabasz(X mat.Matrix, labels []int) float64 {
    score, _ := metrics.CalinskiHarabaszScore(X, labels)
    return score
}
```

### External Metrics

```go
// When true labels are known
func evaluateWithTrueLabels(trueLabels, predLabels []int) {
    // Adjusted Rand Index
    ari := metrics.AdjustedRandIndex(trueLabels, predLabels)
    fmt.Printf("ARI: %.3f\n", ari)
    
    // Normalized Mutual Information
    nmi := metrics.NormalizedMutualInfo(trueLabels, predLabels)
    fmt.Printf("NMI: %.3f\n", nmi)
    
    // Fowlkes-Mallows Score
    fmi := metrics.FowlkesMallows(trueLabels, predLabels)
    fmt.Printf("FMI: %.3f\n", fmi)
}
```

## Feature Engineering for Clustering

### Dimensionality Reduction

```go
func preprocessForClustering(X mat.Matrix) mat.Matrix {
    // Standardize features
    scaler := preprocessing.NewStandardScaler()
    XScaled, _ := scaler.FitTransform(X)
    
    // Apply PCA
    pca := decomposition.NewPCA(
        decomposition.WithNComponents(50),
        decomposition.WithWhiten(true),
    )
    XReduced, _ := pca.FitTransform(XScaled)
    
    return XReduced
}
```

### Feature Scaling

```go
// Clustering is sensitive to scale
func scaleFeatures(X mat.Matrix) mat.Matrix {
    // Use RobustScaler for outliers
    scaler := preprocessing.NewRobustScaler()
    XScaled, _ := scaler.FitTransform(X)
    
    return XScaled
}
```

## Advanced Techniques

### Spectral Clustering

```go
type SpectralClustering struct {
    nClusters int
    affinity  string
    gamma     float64
}

func (sc *SpectralClustering) Fit(X mat.Matrix) []int {
    // Build affinity matrix
    A := sc.buildAffinityMatrix(X)
    
    // Compute Laplacian
    L := sc.computeLaplacian(A)
    
    // Eigendecomposition
    eigenvectors := sc.computeEigenvectors(L, sc.nClusters)
    
    // Cluster eigenvectors using K-Means
    kmeans := cluster.NewKMeans(sc.nClusters)
    kmeans.Fit(eigenvectors)
    
    return kmeans.Labels()
}

func (sc *SpectralClustering) buildAffinityMatrix(X mat.Matrix) *mat.Dense {
    rows, _ := X.Dims()
    A := mat.NewDense(rows, rows, nil)
    
    for i := 0; i < rows; i++ {
        for j := i + 1; j < rows; j++ {
            // RBF kernel
            dist := euclideanDistance(
                X.(*mat.Dense).RowView(i),
                X.(*mat.Dense).RowView(j),
            )
            affinity := math.Exp(-sc.gamma * dist * dist)
            
            A.Set(i, j, affinity)
            A.Set(j, i, affinity)
        }
    }
    
    return A
}
```

### Mean Shift

```go
type MeanShift struct {
    bandwidth float64
    maxIter   int
}

func (ms *MeanShift) Fit(X mat.Matrix) []int {
    rows, cols := X.Dims()
    
    // Initialize seeds
    seeds := mat.DenseCopyOf(X)
    
    // Shift each seed
    for iter := 0; iter < ms.maxIter; iter++ {
        converged := true
        
        for i := 0; i < rows; i++ {
            seed := seeds.RowView(i).(*mat.VecDense)
            
            // Calculate mean shift
            shift := ms.calculateMeanShift(X, seed)
            
            // Update seed
            for j := 0; j < cols; j++ {
                oldVal := seed.AtVec(j)
                newVal := oldVal + shift.AtVec(j)
                seed.SetVec(j, newVal)
                
                if math.Abs(newVal-oldVal) > 1e-4 {
                    converged = false
                }
            }
        }
        
        if converged {
            break
        }
    }
    
    // Merge nearby seeds
    return ms.mergeClusters(seeds)
}
```

## Performance Optimization

### Parallel Clustering

```go
func parallelKMeans(X mat.Matrix, k int, nWorkers int) *KMeans {
    // Split data
    chunks := splitData(X, nWorkers)
    results := make(chan *KMeans, nWorkers)
    
    // Run K-Means on each chunk
    for _, chunk := range chunks {
        go func(data mat.Matrix) {
            km := cluster.NewKMeans(k)
            km.Fit(data)
            results <- km
        }(chunk)
    }
    
    // Merge results
    var allCenters []mat.Vector
    for i := 0; i < nWorkers; i++ {
        km := <-results
        centers := km.ClusterCenters()
        for j := 0; j < k; j++ {
            allCenters = append(allCenters, centers.(*mat.Dense).RowView(j))
        }
    }
    
    // Final clustering on centers
    centerMatrix := vectorsToMatrix(allCenters)
    finalKM := cluster.NewKMeans(k)
    finalKM.Fit(centerMatrix)
    
    return finalKM
}
```

### Approximate Nearest Neighbors

```go
// Use LSH for fast neighbor search
type LSHIndex struct {
    tables     []map[uint64][]int
    hashFuncs  []HashFunc
    nTables    int
}

func (lsh *LSHIndex) FindNeighbors(query mat.Vector, k int) []int {
    candidates := make(map[int]bool)
    
    // Query each table
    for i, table := range lsh.tables {
        hash := lsh.hashFuncs[i](query)
        if indices, exists := table[hash]; exists {
            for _, idx := range indices {
                candidates[idx] = true
            }
        }
    }
    
    // Exact distance calculation on candidates
    return lsh.rankCandidates(query, candidates, k)
}
```

## Best Practices

1. **Scale Features**: Clustering is sensitive to feature scales
2. **Choose Right Algorithm**: Consider data shape and size
3. **Validate Clusters**: Use multiple metrics
4. **Try Multiple K**: Don't assume optimal k
5. **Handle Outliers**: Consider DBSCAN for noisy data
6. **Reduce Dimensions**: For high-dimensional data
7. **Initialize Well**: Use k-means++ or multiple runs
8. **Monitor Convergence**: Check iteration counts

## Common Pitfalls

1. **Wrong Distance Metric**: Euclidean isn't always appropriate
2. **Ignoring Scale**: Features with large ranges dominate
3. **Fixed K**: Real data might not have clear clusters
4. **Overfitting**: Too many clusters for small datasets
5. **Initialization**: Poor seeds lead to suboptimal results

## Next Steps

- Explore [Preprocessing Guide](./preprocessing.md)
- Learn about [Model Persistence](./model-persistence.md)
- See [Clustering Examples](../../examples/clustering/)
- Read [API Reference](../api/sklearn.md#clustering)