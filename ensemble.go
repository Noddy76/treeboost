// Copyright 2026 James Grant
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package treeboost

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"
)

// DefaultMaxLeafValue defines the default ceiling on individual tree leaf predictions (1e4).
const DefaultMaxLeafValue = 1e4

func resolveMaxLeafValue(val float64) float64 {
	if val <= 0.0 || math.IsNaN(val) || math.IsInf(val, 0) {
		return DefaultMaxLeafValue
	}
	return val
}

func clampLeafValue(val float64, maxLeaf float64) float64 {
	if maxLeaf <= 0.0 || math.IsNaN(maxLeaf) || math.IsInf(maxLeaf, 0) {
		maxLeaf = DefaultMaxLeafValue
	}
	if math.IsNaN(val) {
		return 0.0
	}
	if math.IsInf(val, 1) || val > maxLeaf {
		return maxLeaf
	}
	if math.IsInf(val, -1) || val < -maxLeaf {
		return -maxLeaf
	}
	return val
}

// CatBoostParams holds hyperparameters for Symmetric GBDT (obliviated trees).
// Reference: https://en.wikipedia.org/wiki/CatBoost
type CatBoostParams struct {
	Depth                 int             `json:"depth"`
	Iterations            int             `json:"iterations"`
	LearningRate          float64         `json:"learning_rate"`
	L2LeafReg             float64         `json:"l2_leaf_reg"`
	MaxLeafValue          float64         `json:"max_leaf_value,omitempty"`
	LossFunction          string          `json:"loss_function,omitempty"`
	Quantile              float64         `json:"quantile,omitempty"`
	ColsampleByTree       float64         `json:"colsample_bytree,omitempty"`
	ColsampleByNode       float64         `json:"colsample_bynode,omitempty"`
	EarlyStoppingPatience int             `json:"early_stopping_patience,omitempty"`
	ValidationData        ColumnarDataset `json:"-"`
	ValidationX           []FeatureVector `json:"-"`
	ValidationY           []float64       `json:"-"`
	ValidationW           []float64       `json:"-"`
}

// LightGBMParams holds hyperparameters for Leaf-wise GBDT (best-first tree growth).
// Reference: https://en.wikipedia.org/wiki/LightGBM
type LightGBMParams struct {
	MaxDepth              int             `json:"max_depth"`
	MaxLeaves             int             `json:"max_leaves"`
	MinChildSamples       int             `json:"min_child_samples"`
	Estimators            int             `json:"estimators"`
	LearningRate          float64         `json:"learning_rate"`
	L2LeafReg             float64         `json:"l2_leaf_reg"` // L2 leaf regularization parameter (lambda >= 0.0)
	MaxLeafValue          float64         `json:"max_leaf_value,omitempty"`
	LossFunction          string          `json:"loss_function,omitempty"`
	Quantile              float64         `json:"quantile,omitempty"`
	ColsampleByTree       float64         `json:"colsample_bytree,omitempty"`
	ColsampleByNode       float64         `json:"colsample_bynode,omitempty"`
	EarlyStoppingPatience int             `json:"early_stopping_patience,omitempty"`
	ValidationData        ColumnarDataset `json:"-"`
	ValidationX           []FeatureVector `json:"-"`
	ValidationY           []float64       `json:"-"`
	ValidationW           []float64       `json:"-"`
}

// ExtraTreesParams holds hyperparameters for Extremely Randomized Trees.
// Reference: https://en.wikipedia.org/wiki/Random_forest#ExtraTrees
type ExtraTreesParams struct {
	Estimators     int     `json:"estimators"`
	MinSamplesLeaf int     `json:"min_samples_leaf"`
	MaxFeatures    float64 `json:"max_features"`
	MaxLeafValue   float64 `json:"max_leaf_value,omitempty"`
	LossFunction   string  `json:"loss_function,omitempty"`
	Quantile       float64 `json:"quantile,omitempty"`
}

// EnsembleConfig specifies individual algorithm parameters and weighted contribution ratios.
type EnsembleConfig struct {
	MaxLeafValue          float64          `json:"max_leaf_value,omitempty"`
	CatBoostWeight        float64          `json:"catboost_weight"`
	LightGBMWeight        float64          `json:"lightgbm_weight"`
	ExtraTreesWeight      float64          `json:"extratrees_weight"`
	LossFunction          string           `json:"loss_function,omitempty"`
	Quantile              float64          `json:"quantile,omitempty"`
	ColsampleByTree       float64          `json:"colsample_bytree,omitempty"`
	ColsampleByNode       float64          `json:"colsample_bynode,omitempty"`
	EarlyStoppingPatience int              `json:"early_stopping_patience,omitempty"`
	ValidationData        ColumnarDataset  `json:"-"`
	ValidationX           []FeatureVector  `json:"-"`
	ValidationY           []float64        `json:"-"`
	ValidationW           []float64        `json:"-"`
	CatBoost              CatBoostParams   `json:"catboost"`
	LightGBM              LightGBMParams   `json:"lightgbm"`
	ExtraTrees            ExtraTreesParams `json:"extratrees"`
}

// DefaultEnsembleConfig returns a balanced, standard default configuration combining
// Symmetric GBDT, Leaf-wise GBDT, and ExtraTrees into an equal-weight ensemble.
func DefaultEnsembleConfig() EnsembleConfig {
	return EnsembleConfig{
		MaxLeafValue:     DefaultMaxLeafValue,
		CatBoostWeight:   1.0 / 3.0,
		LightGBMWeight:   1.0 / 3.0,
		ExtraTreesWeight: 1.0 / 3.0,
		CatBoost: CatBoostParams{
			Depth:        6,
			Iterations:   500,
			LearningRate: 0.05,
			L2LeafReg:    3.0,
		},
		LightGBM: LightGBMParams{
			MaxDepth:        5,
			MaxLeaves:       31,
			MinChildSamples: 20,
			Estimators:      500,
			LearningRate:    0.05,
			L2LeafReg:       1.0,
		},
		ExtraTrees: ExtraTreesParams{
			Estimators:     700,
			MinSamplesLeaf: 4,
			MaxFeatures:    1.0,
		},
	}
}

// DefaultQuantileEnsembleConfig returns an ensemble configuration optimized for quantile regression,
// setting CatBoostWeight: 0.5, LightGBMWeight: 0.5, ExtraTreesWeight: 0.0, and ExtraTrees.Estimators: 0.
func DefaultQuantileEnsembleConfig(quantile float64) EnsembleConfig {
	validateQuantileParams(LossPinball, quantile)
	cfg := DefaultEnsembleConfig()
	cfg.LossFunction = LossPinball
	cfg.Quantile = quantile
	cfg.CatBoostWeight = 0.5
	cfg.LightGBMWeight = 0.5
	cfg.ExtraTreesWeight = 0.0
	cfg.CatBoost.LossFunction = LossPinball
	cfg.CatBoost.Quantile = quantile
	cfg.LightGBM.LossFunction = LossPinball
	cfg.LightGBM.Quantile = quantile
	cfg.ExtraTrees.LossFunction = LossPinball
	cfg.ExtraTrees.Quantile = quantile
	cfg.ExtraTrees.Estimators = 0
	return cfg
}

type nodeRange struct {
	start int
	end   int
}

// TrainWorkspace holds reusable slice buffers to eliminate heap allocations during tree construction.
type TrainWorkspace struct {
	SampleIdxs      []int
	PermBuf         []int
	SplitBuf        []float64
	Ranges          []nodeRange
	Hist            FeatureHistogram
	QuantileScratch []int
	TreeFeatureIdxs []int
	NodeFeatureIdxs []int
}

var trainWorkspacePool = sync.Pool{
	New: func() interface{} {
		return &TrainWorkspace{
			SampleIdxs:      make([]int, 0, 4096),
			PermBuf:         make([]int, 0, 256),
			SplitBuf:        make([]float64, 50),
			Ranges:          make([]nodeRange, 0, 256),
			QuantileScratch: make([]int, 0, 4096),
			TreeFeatureIdxs: make([]int, 0, 256),
			NodeFeatureIdxs: make([]int, 0, 256),
		}
	},
}

func acquireWorkspace(nSamples, nFeatures int) *TrainWorkspace {
	ws := trainWorkspacePool.Get().(*TrainWorkspace)
	if cap(ws.SampleIdxs) < nSamples {
		ws.SampleIdxs = make([]int, nSamples)
	} else {
		ws.SampleIdxs = ws.SampleIdxs[:nSamples]
	}
	for i := 0; i < nSamples; i++ {
		ws.SampleIdxs[i] = i
	}
	if cap(ws.PermBuf) < nFeatures {
		ws.PermBuf = make([]int, nFeatures)
	} else {
		ws.PermBuf = ws.PermBuf[:nFeatures]
	}
	for i := 0; i < nFeatures; i++ {
		ws.PermBuf[i] = i
	}
	if cap(ws.SplitBuf) < 50 {
		ws.SplitBuf = make([]float64, 50)
	} else {
		ws.SplitBuf = ws.SplitBuf[:50]
	}
	if cap(ws.QuantileScratch) < nSamples {
		ws.QuantileScratch = make([]int, 0, nSamples)
	} else {
		ws.QuantileScratch = ws.QuantileScratch[:0]
	}
	if cap(ws.TreeFeatureIdxs) < nFeatures {
		ws.TreeFeatureIdxs = make([]int, nFeatures)
	} else {
		ws.TreeFeatureIdxs = ws.TreeFeatureIdxs[:nFeatures]
	}
	if cap(ws.NodeFeatureIdxs) < nFeatures {
		ws.NodeFeatureIdxs = make([]int, nFeatures)
	} else {
		ws.NodeFeatureIdxs = ws.NodeFeatureIdxs[:nFeatures]
	}
	return ws
}

func releaseWorkspace(ws *TrainWorkspace) {
	if ws == nil {
		return
	}
	trainWorkspacePool.Put(ws)
}

// validateColsample ensures colsample ratios are strictly valid numbers in [0.0, 1.0].
func validateColsample(name string, val float64) {
	if math.IsNaN(val) || math.IsInf(val, 0) || val < 0.0 || val > 1.0 {
		panic(fmt.Sprintf("treeboost: invalid %s %v; must be finite and in [0.0, 1.0]", name, val))
	}
}

// resolveColsample returns 1.0 if val is 0.0 (or default), validating bounds.
func resolveColsample(name string, val float64) float64 {
	validateColsample(name, val)
	if val <= 0.0 {
		return 1.0
	}
	return val
}

// validateEarlyStopping ensures patience is non-negative.
func validateEarlyStopping(patience int) {
	if patience < 0 {
		panic(fmt.Sprintf("treeboost: invalid early stopping patience %d; must be >= 0", patience))
	}
}

// sampleFeatures selects a subset of features from source without heap allocations.
// If ratio >= 1.0 or source has length <= 1, all source features are copied to dest.
// Otherwise, exactly ceil(ratio * len(source)) unique features are selected using Fisher-Yates shuffle.
func sampleFeatures(source []int, ratio float64, dest []int, permBuf []int, rng *rand.Rand) []int {
	n := len(source)
	if n == 0 {
		return dest[:0]
	}
	if ratio >= 1.0 || ratio <= 0.0 {
		if cap(dest) < n {
			dest = make([]int, n)
		} else {
			dest = dest[:n]
		}
		copy(dest, source)
		return dest
	}
	k := int(math.Ceil(ratio * float64(n)))
	if k < 1 {
		k = 1
	}
	if k > n {
		k = n
	}
	if cap(permBuf) < n {
		permBuf = make([]int, n)
	} else {
		permBuf = permBuf[:n]
	}
	copy(permBuf, source)

	// Partial Fisher-Yates shuffle of permBuf up to k elements
	for i := 0; i < k; i++ {
		j := i + rng.Intn(n-i)
		permBuf[i], permBuf[j] = permBuf[j], permBuf[i]
	}
	if cap(dest) < k {
		dest = make([]int, k)
	} else {
		dest = dest[:k]
	}
	copy(dest, permBuf[:k])
	return dest
}

// partitionSamplesInPlace partitions sampleIdxs[start:end] in-place around splitVal for feature f.
// Elements with feature < splitVal are placed in sampleIdxs[start:mid],
// and elements with feature >= splitVal in sampleIdxs[mid:end].
// Returns mid.
func partitionSamplesInPlace(sampleIdxs []int, start, end int, featureValues []float64, fOffset int, splitVal float64) int {
	i := start
	j := end - 1
	for i <= j {
		for i <= j {
			v := featureValues[fOffset+sampleIdxs[i]]
			if !math.IsNaN(v) && v < splitVal {
				i++
			} else {
				break
			}
		}
		for i <= j {
			v := featureValues[fOffset+sampleIdxs[j]]
			if math.IsNaN(v) || v >= splitVal {
				j--
			} else {
				break
			}
		}
		if i < j {
			sampleIdxs[i], sampleIdxs[j] = sampleIdxs[j], sampleIdxs[i]
			i++
			j--
		}
	}
	return i
}

func getSplitCandidatesColumnar(data ColumnarDataset, indices []int, f int, ws *TrainWorkspace) []float64 {
	if len(indices) == 0 || data.NumSamples == 0 {
		return nil
	}
	fOffset := f * data.NumSamples
	minVal := data.Features[fOffset+indices[0]]
	maxVal := minVal
	for _, idx := range indices {
		v := data.Features[fOffset+idx]
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	if minVal == maxVal {
		return nil
	}
	if ws == nil || cap(ws.SplitBuf) < 50 {
		splits := make([]float64, 50)
		step := (maxVal - minVal) / 51.0
		for i := 0; i < 50; i++ {
			splits[i] = minVal + step*float64(i+1)
		}
		return splits
	}
	splits := ws.SplitBuf[:50]
	step := (maxVal - minVal) / 51.0
	for i := 0; i < 50; i++ {
		splits[i] = minVal + step*float64(i+1)
	}
	return splits
}

func getSplitCandidates(X []FeatureVector, indices []int, f int, ws *TrainWorkspace) []float64 {
	data := ToColumnar(X, nil, nil)
	return getSplitCandidatesColumnar(data, indices, f, ws)
}

func computeWeightedError(Y []float64, W []float64, indices []int) (float64, float64, float64) {
	if len(indices) == 0 {
		return 0.0, 0.0, 0.0
	}
	sumW, sumWY := computeWeightedSum(W, Y, indices)
	if sumW <= 1e-9 {
		return 0.0, 0.0, 0.0
	}
	mean := sumWY / sumW
	var errorSq float64
	for _, idx := range indices {
		diff := Y[idx] - mean
		errorSq += W[idx] * diff * diff
	}
	return errorSq, mean, sumW
}

type splitCandidateResult struct {
	feature int
	val     float64
	score   float64
}

func evaluateSymmetricFeatureSplits(
	data ColumnarDataset,
	f int,
	levelStart, levelEnd int,
	ranges []nodeRange,
	sampleIdxs []int,
	ws *TrainWorkspace,
) splitCandidateResult {
	res := splitCandidateResult{feature: -1, val: 0.0, score: math.MaxFloat64}
	fOffset := f * data.NumSamples

	var candidates []float64
	if len(data.Binned.Bins) > 0 {
		BuildHistogramSlice(data.Binned.Bins, fOffset, data.Weights, data.Targets, sampleIdxs, 0, data.NumSamples, &ws.Hist)
		totalW, totalWY := computeWeightedSumContiguous(data.Weights, data.Targets)
		val, _, ok := FindBestHistogramSplit(&ws.Hist, data.Binned.BinBoundaries[f], totalW, totalWY)
		if ok {
			candidates = []float64{val}
		}
	} else {
		candidates = getSplitCandidatesColumnar(data, sampleIdxs[:data.NumSamples], f, ws)
	}
	if len(candidates) == 0 {
		return res
	}

	for _, val := range candidates {
		var totalScore float64
		for nIdx := levelStart; nIdx < levelEnd; nIdx++ {
			r := ranges[nIdx]
			if r.end-r.start == 0 {
				continue
			}
			var sumWLeft, sumWYLeft, sumWY2Left float64
			var sumWRight, sumWYRight, sumWY2Right float64
			for i := r.start; i < r.end; i++ {
				idx := sampleIdxs[i]
				w := data.Weights[idx]
				y := data.Targets[idx]
				if data.Features[fOffset+idx] < val {
					sumWLeft += w
					sumWYLeft += w * y
					sumWY2Left += w * y * y
				} else {
					sumWRight += w
					sumWYRight += w * y
					sumWY2Right += w * y * y
				}
			}
			var errLeft, errRight float64
			if sumWLeft > 1e-9 {
				errLeft = sumWY2Left - (sumWYLeft*sumWYLeft)/sumWLeft
				if errLeft < 0 || math.IsNaN(errLeft) || math.IsInf(errLeft, 0) {
					errLeft = 0
				}
			}
			if sumWRight > 1e-9 {
				errRight = sumWY2Right - (sumWYRight*sumWYRight)/sumWRight
				if errRight < 0 || math.IsNaN(errRight) || math.IsInf(errRight, 0) {
					errRight = 0
				}
			}
			totalScore += errLeft + errRight
		}

		if math.IsNaN(totalScore) || math.IsInf(totalScore, 0) {
			continue
		}

		if totalScore < res.score {
			res.score = totalScore
			res.feature = f
			res.val = val
		}
	}
	return res
}

func trainSymmetricTreeColumnarWithColsample(ws *TrainWorkspace, data ColumnarDataset, depth int, lambda float64, maxLeaf float64, lossFunction string, quantile float64, residuals []float64, colsampleByNode float64, treeFeatures []int, rng *rand.Rand) []Node {
	if depth < 0 {
		depth = 0
	} else if depth > 16 {
		depth = 16
	}
	if lambda < 0.0 || math.IsNaN(lambda) || math.IsInf(lambda, 0) {
		lambda = 0.0
	}
	if data.NumSamples > 0 && len(data.Weights) == 0 {
		weights := make([]float64, data.NumSamples)
		for i := range weights {
			weights[i] = 1.0
		}
		data.Weights = weights
	}
	numNodes := (1 << (depth + 1)) - 1
	nodes := make([]Node, numNodes)
	for i := range nodes {
		nodes[i] = Node{NodeID: i, SplitFeature: -1}
	}

	if cap(ws.Ranges) < numNodes {
		ws.Ranges = make([]nodeRange, numNodes)
	} else {
		ws.Ranges = ws.Ranges[:numNodes]
	}

	for i := 0; i < data.NumSamples; i++ {
		ws.SampleIdxs[i] = i
	}
	ws.Ranges[0] = nodeRange{start: 0, end: data.NumSamples}

	// Prepare tree-level feature candidates
	activeFeatures := treeFeatures
	if len(activeFeatures) == 0 {
		if cap(ws.PermBuf) < data.NumFeatures {
			ws.PermBuf = make([]int, data.NumFeatures)
		} else {
			ws.PermBuf = ws.PermBuf[:data.NumFeatures]
		}
		for i := 0; i < data.NumFeatures; i++ {
			ws.PermBuf[i] = i
		}
		activeFeatures = ws.PermBuf
	}

	actualDepth := depth
	for d := 0; d < depth; d++ {
		levelStart := (1 << d) - 1
		levelEnd := (1 << (d + 1)) - 1

		// Subsample features per node/level if colsampleByNode < 1.0
		evalFeatures := activeFeatures
		if colsampleByNode > 0.0 && colsampleByNode < 1.0 && rng != nil {
			evalFeatures = sampleFeatures(activeFeatures, colsampleByNode, ws.NodeFeatureIdxs, ws.PermBuf, rng)
		}

		bestFeature := -1
		bestVal := 0.0
		bestScore := math.MaxFloat64

		numFeatures := len(evalFeatures)
		concurrency := runtime.GOMAXPROCS(0)
		if numFeatures >= 8 && concurrency > 1 {
			// Intra-tree parallel split search across features
			numWorkers := concurrency
			if numWorkers > numFeatures {
				numWorkers = numFeatures
			}
			chunkSize := (numFeatures + numWorkers - 1) / numWorkers
			results := make([]splitCandidateResult, numWorkers)
			var wg sync.WaitGroup

			for w := 0; w < numWorkers; w++ {
				fStart := w * chunkSize
				fEnd := fStart + chunkSize
				if fStart >= numFeatures {
					results[w] = splitCandidateResult{feature: -1, score: math.MaxFloat64}
					continue
				}
				if fEnd > numFeatures {
					fEnd = numFeatures
				}

				wg.Add(1)
				go func(workerID, startIdx, endIdx int) {
					defer wg.Done()
					tws := acquireWorkspace(data.NumSamples, data.NumFeatures)
					defer releaseWorkspace(tws)

					workerBest := splitCandidateResult{feature: -1, score: math.MaxFloat64}
					for idx := startIdx; idx < endIdx; idx++ {
						feat := evalFeatures[idx]
						candRes := evaluateSymmetricFeatureSplits(data, feat, levelStart, levelEnd, ws.Ranges, ws.SampleIdxs, tws)
						if candRes.feature != -1 && candRes.score < workerBest.score {
							workerBest = candRes
						}
					}
					results[workerID] = workerBest
				}(w, fStart, fEnd)
			}
			wg.Wait()

			for _, res := range results {
				if res.feature != -1 && res.score < bestScore {
					bestScore = res.score
					bestFeature = res.feature
					bestVal = res.val
				}
			}
		} else {
			// Sequential evaluation on caller's workspace
			for _, f := range evalFeatures {
				candRes := evaluateSymmetricFeatureSplits(data, f, levelStart, levelEnd, ws.Ranges, ws.SampleIdxs, ws)
				if candRes.feature != -1 && candRes.score < bestScore {
					bestScore = candRes.score
					bestFeature = candRes.feature
					bestVal = candRes.val
				}
			}
		}

		if bestFeature == -1 {
			actualDepth = d
			break
		}

		fOffset := bestFeature * data.NumSamples
		for nIdx := levelStart; nIdx < levelEnd; nIdx++ {
			nodes[nIdx].SplitFeature = bestFeature
			nodes[nIdx].SplitValue = bestVal
			nodes[nIdx].LeftChild = 2*nIdx + 1
			nodes[nIdx].RightChild = 2*nIdx + 2

			r := ws.Ranges[nIdx]
			if r.end-r.start == 0 {
				ws.Ranges[2*nIdx+1] = nodeRange{start: r.start, end: r.start}
				ws.Ranges[2*nIdx+2] = nodeRange{start: r.start, end: r.start}
				continue
			}

			mid := partitionSamplesInPlace(ws.SampleIdxs, r.start, r.end, data.Features, fOffset, bestVal)
			ws.Ranges[2*nIdx+1] = nodeRange{start: r.start, end: mid}
			ws.Ranges[2*nIdx+2] = nodeRange{start: mid, end: r.end}
		}
	}

	leafStart := (1 << actualDepth) - 1
	leafEnd := (1 << (actualDepth + 1)) - 1
	nodes = nodes[:leafEnd]
	for nIdx := leafStart; nIdx < leafEnd; nIdx++ {
		r := ws.Ranges[nIdx]
		nodes[nIdx].SplitFeature = -1
		if r.end-r.start == 0 {
			nodes[nIdx].LeafValue = 0.0
			continue
		}
		if lossFunction == LossPinball && len(residuals) > 0 {
			var sumW float64
			for i := r.start; i < r.end; i++ {
				idx := ws.SampleIdxs[i]
				sumW += data.Weights[idx]
			}
			leafVal := 0.0
			if sumW+lambda > 1e-9 {
				qVal := weightedQuantileIndexed(ws.SampleIdxs[r.start:r.end], residuals, data.Weights, quantile, &ws.QuantileScratch)
				leafVal = (sumW / (sumW + lambda)) * qVal
			}
			nodes[nIdx].LeafValue = clampLeafValue(leafVal, maxLeaf)
		} else {
			var sumWY, sumW float64
			for i := r.start; i < r.end; i++ {
				idx := ws.SampleIdxs[i]
				sumWY += data.Weights[idx] * data.Targets[idx]
				sumW += data.Weights[idx]
			}
			leafVal := 0.0
			if sumW+lambda > 1e-9 {
				leafVal = sumWY / (sumW + lambda)
			}
			nodes[nIdx].LeafValue = clampLeafValue(leafVal, maxLeaf)
		}
	}

	var prune func(curr int)
	prune = func(curr int) {
		if curr >= len(nodes) {
			return
		}
		node := &nodes[curr]
		if node.SplitFeature == -1 {
			return
		}
		prune(node.LeftChild)
		prune(node.RightChild)
		if node.LeftChild < len(nodes) && node.RightChild < len(nodes) {
			leftNode := nodes[node.LeftChild]
			rightNode := nodes[node.RightChild]
			if leftNode.SplitFeature == -1 && rightNode.SplitFeature == -1 && leftNode.LeafValue == 0.0 && rightNode.LeafValue == 0.0 {
				node.SplitFeature = -1
				node.LeafValue = 0.0
			}
		}
	}
	prune(0)

	return nodes
}

func trainSymmetricTreeColumnar(ws *TrainWorkspace, data ColumnarDataset, depth int, lambda float64, maxLeaf float64, lossFunction string, quantile float64, residuals []float64) []Node {
	return trainSymmetricTreeColumnarWithColsample(ws, data, depth, lambda, maxLeaf, lossFunction, quantile, residuals, 1.0, nil, nil)
}

func trainSymmetricTree(ws *TrainWorkspace, X []FeatureVector, Y []float64, W []float64, depth int, lambda float64, maxLeaf float64) []Node {
	data := ToColumnar(X, Y, W)
	return trainSymmetricTreeColumnar(ws, data, depth, lambda, maxLeaf, "", 0.0, nil)
}

// resolveValidationColumnar prepares a validation ColumnarDataset from either params.ValidationData or params.ValidationX/Y/W.
func resolveValidationColumnar(data ColumnarDataset, valData ColumnarDataset, valX []FeatureVector, valY []float64, valW []float64) ColumnarDataset {
	if valData.NumSamples > 0 && len(valData.Targets) == valData.NumSamples {
		if len(valData.Weights) == 0 {
			weights := make([]float64, valData.NumSamples)
			for i := range weights {
				weights[i] = 1.0
			}
			valData.Weights = weights
		}
		return valData
	}
	if len(valX) > 0 && len(valY) == len(valX) {
		vd := ToColumnar(valX, valY, valW)
		if len(vd.Weights) == 0 && vd.NumSamples > 0 {
			weights := make([]float64, vd.NumSamples)
			for i := range weights {
				weights[i] = 1.0
			}
			vd.Weights = weights
		}
		return vd
	}
	return ColumnarDataset{NumSamples: 0}
}

// computeValidationLoss evaluates either Pinball loss or MSE over validation predictions.
func computeValidationLoss(valData ColumnarDataset, preds []float64, isPinball bool, quantile float64) float64 {
	n := valData.NumSamples
	if n == 0 || len(preds) != n {
		return 0.0
	}
	if isPinball {
		return PinballLoss(valData.Targets, preds, valData.Weights, quantile)
	}
	var totalW, sse float64
	for i := 0; i < n; i++ {
		w := 1.0
		if len(valData.Weights) > i {
			w = valData.Weights[i]
		}
		diff := valData.Targets[i] - preds[i]
		sse += w * diff * diff
		totalW += w
	}
	if totalW <= 1e-9 {
		return 0.0
	}
	return sse / totalW
}

// TrainSymmetricGBDTColumnar fits a CatBoost-style GBDT on ColumnarDataset without per-iteration conversions.
func TrainSymmetricGBDTColumnar(data ColumnarDataset, params CatBoostParams) ([]Tree, float64) {
	if params.LossFunction == LossPinball {
		validateQuantileParams(params.LossFunction, params.Quantile)
	}
	validateEarlyStopping(params.EarlyStoppingPatience)
	colsampleByTree := resolveColsample("colsample_bytree", params.ColsampleByTree)
	colsampleByNode := resolveColsample("colsample_bynode", params.ColsampleByNode)

	n := data.NumSamples
	if n == 0 {
		return nil, 0.0
	}
	if len(data.Targets) != n {
		panic(fmt.Sprintf("treeboost: length mismatch: len(Targets)=%d != NumSamples=%d", len(data.Targets), n))
	}
	if len(data.Weights) > 0 && len(data.Weights) != n {
		panic(fmt.Sprintf("treeboost: length mismatch: len(Weights)=%d != NumSamples=%d", len(data.Weights), n))
	}
	if len(data.Weights) == 0 {
		weights := make([]float64, n)
		for i := range weights {
			weights[i] = 1.0
		}
		data.Weights = weights
	}
	maxLeaf := resolveMaxLeafValue(params.MaxLeafValue)
	baseValue := 0.0
	if params.LossFunction == LossPinball {
		baseValue = WeightedQuantile(data.Targets, data.Weights, params.Quantile)
	} else {
		sumW, sumWY := computeWeightedSumContiguous(data.Weights, data.Targets)
		if sumW > 1e-9 {
			baseValue = sumWY / sumW
		}
	}
	baseValue = clampLeafValue(baseValue, maxLeaf)

	if params.Iterations <= 0 {
		return nil, baseValue
	}

	valData := resolveValidationColumnar(data, params.ValidationData, params.ValidationX, params.ValidationY, params.ValidationW)
	useEarlyStopping := params.EarlyStoppingPatience > 0 && valData.NumSamples > 0

	var valPreds []float64
	bestValLoss := math.MaxFloat64
	bestIter := -1
	patienceCount := 0

	if useEarlyStopping {
		valPreds = make([]float64, valData.NumSamples)
		for i := 0; i < valData.NumSamples; i++ {
			valPreds[i] = baseValue
		}
		bestValLoss = computeValidationLoss(valData, valPreds, params.LossFunction == LossPinball, params.Quantile)
	}

	preds := make([]float64, n)
	for i := 0; i < n; i++ {
		preds[i] = baseValue
	}

	trees := make([]Tree, 0, params.Iterations)
	residuals := make([]float64, n)
	var gradients []float64
	isPinball := params.LossFunction == LossPinball
	if isPinball {
		gradients = make([]float64, n)
	}

	ws := acquireWorkspace(data.NumSamples, data.NumFeatures)
	defer releaseWorkspace(ws)

	rng := rand.New(rand.NewSource(42))
	allFeatures := make([]int, data.NumFeatures)
	for i := 0; i < data.NumFeatures; i++ {
		allFeatures[i] = i
	}

	iterData := data
	for m := 0; m < params.Iterations; m++ {
		for i := 0; i < n; i++ {
			residuals[i] = data.Targets[i] - preds[i]
		}
		if isPinball {
			q := params.Quantile
			for i := 0; i < n; i++ {
				r := residuals[i]
				if r > 0 {
					gradients[i] = q
				} else if r < 0 {
					gradients[i] = q - 1.0
				} else {
					gradients[i] = 0.0
				}
			}
			iterData.Targets = gradients
		} else {
			iterData.Targets = residuals
		}

		// Feature subsampling per tree
		treeFeatures := allFeatures
		if colsampleByTree > 0.0 && colsampleByTree < 1.0 {
			treeFeatures = sampleFeatures(allFeatures, colsampleByTree, ws.TreeFeatureIdxs, ws.PermBuf, rng)
		}

		treeNodes := trainSymmetricTreeColumnarWithColsample(ws, iterData, params.Depth, params.L2LeafReg, maxLeaf, params.LossFunction, params.Quantile, residuals, colsampleByNode, treeFeatures, rng)
		trees = append(trees, Tree{Nodes: treeNodes})

		for i := 0; i < n; i++ {
			preds[i] += params.LearningRate * EvaluateTreeColumnar(treeNodes, iterData, i)
		}

		if useEarlyStopping {
			for i := 0; i < valData.NumSamples; i++ {
				valPreds[i] += params.LearningRate * EvaluateTreeColumnar(treeNodes, valData, i)
			}
			valLoss := computeValidationLoss(valData, valPreds, isPinball, params.Quantile)
			if valLoss < bestValLoss-1e-12 {
				bestValLoss = valLoss
				bestIter = m
				patienceCount = 0
			} else {
				patienceCount++
				if patienceCount >= params.EarlyStoppingPatience {
					if bestIter >= 0 {
						trees = trees[:bestIter+1]
					}
					break
				}
			}
		}
	}

	return trees, baseValue
}

// TrainSymmetricGBDT fits a symmetric (CatBoost-style) gradient boosted decision tree ensemble.
func TrainSymmetricGBDT(X []FeatureVector, Y []float64, W []float64, params CatBoostParams) ([]Tree, float64) {
	if len(X) != len(Y) {
		panic(fmt.Sprintf("treeboost: length mismatch: len(X)=%d != len(Y)=%d", len(X), len(Y)))
	}
	if len(W) > 0 && len(W) != len(Y) {
		panic(fmt.Sprintf("treeboost: length mismatch: len(W)=%d != len(Y)=%d", len(W), len(Y)))
	}
	data := ToColumnar(X, Y, W)
	return TrainSymmetricGBDTColumnar(data, params)
}

type lgbBuildNode struct {
	nodeID       int
	depth        int
	start        int
	end          int
	splitFeature int
	splitValue   float64
	leftChild    int
	rightChild   int
	leafValue    float64
	sumW         float64
	sumWY        float64
}

type leafwiseSplitCandidateResult struct {
	gain      float64
	activeIdx int
	feature   int
	val       float64
	leftMean  float64
	rightMean float64
	leftW     float64
	leftWY    float64
	rightW    float64
	rightWY   float64
}

func evaluateLeafwiseFeatureSplits(
	data ColumnarDataset,
	f int,
	activeLeaves []int,
	buildNodes []lgbBuildNode,
	sampleIdxs []int,
	maxDepth int,
	minChildSamples int,
	lambda float64,
	maxLeaf float64,
	ws *TrainWorkspace,
) leafwiseSplitCandidateResult {
	res := leafwiseSplitCandidateResult{gain: -1.0, activeIdx: -1, feature: -1}
	fOffset := f * data.NumSamples

	for i, nodeIdx := range activeLeaves {
		leaf := &buildNodes[nodeIdx]
		nSamples := leaf.end - leaf.start
		if leaf.depth >= maxDepth || nSamples < minChildSamples*2 {
			continue
		}

		if len(data.Binned.Bins) > 0 {
			BuildHistogramSlice(data.Binned.Bins, fOffset, data.Weights, data.Targets, sampleIdxs, leaf.start, leaf.end, &ws.Hist)
			val, _, leftW, leftWY, rightW, rightWY, ok := FindBestHistogramSplitWithCounts(
				&ws.Hist, data.Binned.BinBoundaries[f], leaf.sumW, leaf.sumWY, nSamples, minChildSamples)
			if !ok {
				continue
			}

			parentGain := 0.0
			if leaf.sumW+lambda > 1e-9 {
				parentGain = (leaf.sumWY * leaf.sumWY) / (leaf.sumW + lambda)
			}
			gain := (leftWY*leftWY/(leftW+lambda) + rightWY*rightWY/(rightW+lambda)) - parentGain
			if math.IsNaN(gain) || math.IsInf(gain, 0) {
				continue
			}

			if gain > res.gain {
				res.gain = gain
				res.activeIdx = i
				res.feature = f
				res.val = val
				res.leftMean = clampLeafValue(leftWY/(leftW+lambda), maxLeaf)
				res.rightMean = clampLeafValue(rightWY/(rightW+lambda), maxLeaf)
				res.leftW = leftW
				res.leftWY = leftWY
				res.rightW = rightW
				res.rightWY = rightWY
			}
		} else {
			candidates := getSplitCandidatesColumnar(data, sampleIdxs[leaf.start:leaf.end], f, ws)
			if len(candidates) == 0 {
				continue
			}

			for _, val := range candidates {
				var sumWLeft, sumWYLeft, sumWY2Left float64
				var sumWRight, sumWYRight, sumWY2Right float64
				var countLeft, countRight int
				for s := leaf.start; s < leaf.end; s++ {
					idx := sampleIdxs[s]
					w := data.Weights[idx]
					y := data.Targets[idx]
					if data.Features[fOffset+idx] < val {
						sumWLeft += w
						sumWYLeft += w * y
						sumWY2Left += w * y * y
						countLeft++
					} else {
						sumWRight += w
						sumWYRight += w * y
						sumWY2Right += w * y * y
						countRight++
					}
				}
				if countLeft < minChildSamples || countRight < minChildSamples {
					continue
				}
				if sumWLeft <= 1e-9 || sumWRight <= 1e-9 {
					continue
				}

				parentGain := 0.0
				if leaf.sumW+lambda > 1e-9 {
					parentGain = (leaf.sumWY * leaf.sumWY) / (leaf.sumW + lambda)
				}
				gain := (sumWYLeft*sumWYLeft/(sumWLeft+lambda) + sumWYRight*sumWYRight/(sumWRight+lambda)) - parentGain
				if math.IsNaN(gain) || math.IsInf(gain, 0) {
					continue
				}

				if gain > res.gain {
					res.gain = gain
					res.activeIdx = i
					res.feature = f
					res.val = val
					res.leftMean = clampLeafValue(sumWYLeft/(sumWLeft+lambda), maxLeaf)
					res.rightMean = clampLeafValue(sumWYRight/(sumWRight+lambda), maxLeaf)
					res.leftW = sumWLeft
					res.leftWY = sumWYLeft
					res.rightW = sumWRight
					res.rightWY = sumWYRight
				}
			}
		}
	}
	return res
}

func trainLeafwiseTreeColumnarWithColsample(ws *TrainWorkspace, data ColumnarDataset, maxDepth int, maxLeaves int, minChildSamples int, lambda float64, maxLeaf float64, lossFunction string, quantile float64, residuals []float64, colsampleByNode float64, treeFeatures []int, rng *rand.Rand) []Node {
	if maxDepth < 0 {
		maxDepth = 0
	}
	if maxLeaves < 2 {
		maxLeaves = 2
	}
	if minChildSamples < 1 {
		minChildSamples = 1
	}
	if lambda < 0.0 || math.IsNaN(lambda) || math.IsInf(lambda, 0) {
		lambda = 0.0
	}
	if data.NumSamples > 0 && len(data.Weights) == 0 {
		weights := make([]float64, data.NumSamples)
		for i := range weights {
			weights[i] = 1.0
		}
		data.Weights = weights
	}
	for i := 0; i < data.NumSamples; i++ {
		ws.SampleIdxs[i] = i
	}

	var sumW, sumWY float64
	for i := 0; i < data.NumSamples; i++ {
		w := data.Weights[i]
		y := data.Targets[i]
		sumW += w
		sumWY += w * y
	}

	meanRoot := 0.0
	if sumW+lambda > 1e-9 {
		meanRoot = sumWY / (sumW + lambda)
	}
	meanRoot = clampLeafValue(meanRoot, maxLeaf)

	buildNodes := make([]lgbBuildNode, 0, maxLeaves*2+1)
	buildNodes = append(buildNodes, lgbBuildNode{
		nodeID:       0,
		depth:        0,
		start:        0,
		end:          data.NumSamples,
		splitFeature: -1,
		leftChild:    -1,
		rightChild:   -1,
		leafValue:    meanRoot,
		sumW:         sumW,
		sumWY:        sumWY,
	})

	activeLeaves := []int{0}

	// Prepare tree-level feature candidates
	activeFeatures := treeFeatures
	if len(activeFeatures) == 0 {
		if cap(ws.PermBuf) < data.NumFeatures {
			ws.PermBuf = make([]int, data.NumFeatures)
		} else {
			ws.PermBuf = ws.PermBuf[:data.NumFeatures]
		}
		for i := 0; i < data.NumFeatures; i++ {
			ws.PermBuf[i] = i
		}
		activeFeatures = ws.PermBuf
	}

	for len(activeLeaves) < maxLeaves {
		// Subsample features per node split if colsampleByNode < 1.0
		evalFeatures := activeFeatures
		if colsampleByNode > 0.0 && colsampleByNode < 1.0 && rng != nil {
			evalFeatures = sampleFeatures(activeFeatures, colsampleByNode, ws.NodeFeatureIdxs, ws.PermBuf, rng)
		}

		bestActiveIdx := -1
		bestGain := -1.0
		bestF := -1
		bestVal := 0.0
		var bestLeftMean, bestRightMean float64
		var bestLeftW, bestLeftWY, bestRightW, bestRightWY float64

		numFeatures := len(evalFeatures)
		concurrency := runtime.GOMAXPROCS(0)
		if numFeatures >= 8 && concurrency > 1 {
			// Intra-tree parallel split search across features
			numWorkers := concurrency
			if numWorkers > numFeatures {
				numWorkers = numFeatures
			}
			chunkSize := (numFeatures + numWorkers - 1) / numWorkers
			results := make([]leafwiseSplitCandidateResult, numWorkers)
			var wg sync.WaitGroup

			for w := 0; w < numWorkers; w++ {
				fStart := w * chunkSize
				fEnd := fStart + chunkSize
				if fStart >= numFeatures {
					results[w] = leafwiseSplitCandidateResult{gain: -1.0, activeIdx: -1, feature: -1}
					continue
				}
				if fEnd > numFeatures {
					fEnd = numFeatures
				}

				wg.Add(1)
				go func(workerID, startIdx, endIdx int) {
					defer wg.Done()
					tws := acquireWorkspace(data.NumSamples, data.NumFeatures)
					defer releaseWorkspace(tws)

					workerBest := leafwiseSplitCandidateResult{gain: -1.0, activeIdx: -1, feature: -1}
					for idx := startIdx; idx < endIdx; idx++ {
						feat := evalFeatures[idx]
						candRes := evaluateLeafwiseFeatureSplits(
							data, feat, activeLeaves, buildNodes, ws.SampleIdxs,
							maxDepth, minChildSamples, lambda, maxLeaf, tws,
						)
						if candRes.gain > workerBest.gain {
							workerBest = candRes
						}
					}
					results[workerID] = workerBest
				}(w, fStart, fEnd)
			}
			wg.Wait()

			for _, res := range results {
				if res.gain > bestGain {
					bestGain = res.gain
					bestActiveIdx = res.activeIdx
					bestF = res.feature
					bestVal = res.val
					bestLeftMean = res.leftMean
					bestRightMean = res.rightMean
					bestLeftW = res.leftW
					bestLeftWY = res.leftWY
					bestRightW = res.rightW
					bestRightWY = res.rightWY
				}
			}
		} else {
			// Sequential evaluation on caller's workspace
			for _, f := range evalFeatures {
				candRes := evaluateLeafwiseFeatureSplits(
					data, f, activeLeaves, buildNodes, ws.SampleIdxs,
					maxDepth, minChildSamples, lambda, maxLeaf, ws,
				)
				if candRes.gain > bestGain {
					bestGain = candRes.gain
					bestActiveIdx = candRes.activeIdx
					bestF = candRes.feature
					bestVal = candRes.val
					bestLeftMean = candRes.leftMean
					bestRightMean = candRes.rightMean
					bestLeftW = candRes.leftW
					bestLeftWY = candRes.leftWY
					bestRightW = candRes.rightW
					bestRightWY = candRes.rightWY
				}
			}
		}

		if bestActiveIdx == -1 || bestGain <= 1e-9 {
			break
		}

		parentIdx := activeLeaves[bestActiveIdx]
		parent := &buildNodes[parentIdx]

		mid := partitionSamplesInPlace(ws.SampleIdxs, parent.start, parent.end, data.Features, bestF*data.NumSamples, bestVal)
		if mid == parent.start || mid == parent.end {
			// Partitioning could not divide samples (e.g. constant feature values).
			// Do not corrupt parent node; keep it as a leaf and remove from active list.
			activeLeaves = append(activeLeaves[:bestActiveIdx], activeLeaves[bestActiveIdx+1:]...)
			continue
		}

		parent.splitFeature = bestF
		parent.splitValue = bestVal

		leftNodeID := len(buildNodes)
		rightNodeID := len(buildNodes) + 1

		buildNodes[parentIdx].leftChild = leftNodeID
		buildNodes[parentIdx].rightChild = rightNodeID

		buildNodes = append(buildNodes,
			lgbBuildNode{
				nodeID:       leftNodeID,
				depth:        buildNodes[parentIdx].depth + 1,
				start:        buildNodes[parentIdx].start,
				end:          mid,
				splitFeature: -1,
				leftChild:    -1,
				rightChild:   -1,
				leafValue:    bestLeftMean,
				sumW:         bestLeftW,
				sumWY:        bestLeftWY,
			},
			lgbBuildNode{
				nodeID:       rightNodeID,
				depth:        buildNodes[parentIdx].depth + 1,
				start:        mid,
				end:          buildNodes[parentIdx].end,
				splitFeature: -1,
				leftChild:    -1,
				rightChild:   -1,
				leafValue:    bestRightMean,
				sumW:         bestRightW,
				sumWY:        bestRightWY,
			},
		)

		activeLeaves = append(activeLeaves[:bestActiveIdx], activeLeaves[bestActiveIdx+1:]...)
		activeLeaves = append(activeLeaves, leftNodeID, rightNodeID)
	}

	nodes := make([]Node, len(buildNodes))
	for i, bn := range buildNodes {
		leafVal := bn.leafValue
		if bn.splitFeature == -1 && lossFunction == LossPinball && len(residuals) > 0 {
			if bn.end-bn.start == 0 {
				leafVal = 0.0
			} else {
				var sumW float64
				for j := bn.start; j < bn.end; j++ {
					idx := ws.SampleIdxs[j]
					sumW += data.Weights[idx]
				}
				if sumW+lambda > 1e-9 {
					qVal := weightedQuantileIndexed(ws.SampleIdxs[bn.start:bn.end], residuals, data.Weights, quantile, &ws.QuantileScratch)
					leafVal = (sumW / (sumW + lambda)) * qVal
				} else {
					leafVal = 0.0
				}
			}
		}
		nodes[i] = Node{
			NodeID:       bn.nodeID,
			SplitFeature: bn.splitFeature,
			SplitValue:   bn.splitValue,
			LeafValue:    clampLeafValue(leafVal, maxLeaf),
			LeftChild:    bn.leftChild,
			RightChild:   bn.rightChild,
		}
	}

	return nodes
}

func trainLeafwiseTreeColumnar(ws *TrainWorkspace, data ColumnarDataset, maxDepth int, maxLeaves int, minChildSamples int, lambda float64, maxLeaf float64, lossFunction string, quantile float64, residuals []float64) []Node {
	return trainLeafwiseTreeColumnarWithColsample(ws, data, maxDepth, maxLeaves, minChildSamples, lambda, maxLeaf, lossFunction, quantile, residuals, 1.0, nil, nil)
}

func trainLeafwiseTree(ws *TrainWorkspace, X []FeatureVector, Y []float64, W []float64, maxDepth int, maxLeaves int, minChildSamples int, lambda float64, maxLeaf float64) []Node {
	data := ToColumnar(X, Y, W)
	return trainLeafwiseTreeColumnar(ws, data, maxDepth, maxLeaves, minChildSamples, lambda, maxLeaf, "", 0.0, nil)
}

// TrainLeafwiseGBDTColumnar fits a leaf-wise GBDT on ColumnarDataset without per-iteration conversions.
func TrainLeafwiseGBDTColumnar(data ColumnarDataset, params LightGBMParams) ([]Tree, float64) {
	if params.LossFunction == LossPinball {
		validateQuantileParams(params.LossFunction, params.Quantile)
	}
	validateEarlyStopping(params.EarlyStoppingPatience)
	colsampleByTree := resolveColsample("colsample_bytree", params.ColsampleByTree)
	colsampleByNode := resolveColsample("colsample_bynode", params.ColsampleByNode)

	n := data.NumSamples
	if n == 0 {
		return nil, 0.0
	}
	if len(data.Targets) != n {
		panic(fmt.Sprintf("treeboost: length mismatch: len(Targets)=%d != NumSamples=%d", len(data.Targets), n))
	}
	if len(data.Weights) > 0 && len(data.Weights) != n {
		panic(fmt.Sprintf("treeboost: length mismatch: len(Weights)=%d != NumSamples=%d", len(data.Weights), n))
	}
	if len(data.Weights) == 0 {
		weights := make([]float64, n)
		for i := range weights {
			weights[i] = 1.0
		}
		data.Weights = weights
	}
	maxLeaf := resolveMaxLeafValue(params.MaxLeafValue)
	baseValue := 0.0
	if params.LossFunction == LossPinball {
		baseValue = WeightedQuantile(data.Targets, data.Weights, params.Quantile)
	} else {
		sumW, sumWY := computeWeightedSumContiguous(data.Weights, data.Targets)
		if sumW > 1e-9 {
			baseValue = sumWY / sumW
		}
	}
	baseValue = clampLeafValue(baseValue, maxLeaf)

	if params.Estimators <= 0 {
		return nil, baseValue
	}

	valData := resolveValidationColumnar(data, params.ValidationData, params.ValidationX, params.ValidationY, params.ValidationW)
	useEarlyStopping := params.EarlyStoppingPatience > 0 && valData.NumSamples > 0

	var valPreds []float64
	bestValLoss := math.MaxFloat64
	bestIter := -1
	patienceCount := 0

	if useEarlyStopping {
		valPreds = make([]float64, valData.NumSamples)
		for i := 0; i < valData.NumSamples; i++ {
			valPreds[i] = baseValue
		}
		bestValLoss = computeValidationLoss(valData, valPreds, params.LossFunction == LossPinball, params.Quantile)
	}

	preds := make([]float64, n)
	for i := 0; i < n; i++ {
		preds[i] = baseValue
	}

	trees := make([]Tree, 0, params.Estimators)
	residuals := make([]float64, n)
	var gradients []float64
	isPinball := params.LossFunction == LossPinball
	if isPinball {
		gradients = make([]float64, n)
	}

	ws := acquireWorkspace(data.NumSamples, data.NumFeatures)
	defer releaseWorkspace(ws)

	rng := rand.New(rand.NewSource(42))
	allFeatures := make([]int, data.NumFeatures)
	for i := 0; i < data.NumFeatures; i++ {
		allFeatures[i] = i
	}

	iterData := data
	for m := 0; m < params.Estimators; m++ {
		for i := 0; i < n; i++ {
			residuals[i] = data.Targets[i] - preds[i]
		}
		if isPinball {
			q := params.Quantile
			for i := 0; i < n; i++ {
				r := residuals[i]
				if r > 0 {
					gradients[i] = q
				} else if r < 0 {
					gradients[i] = q - 1.0
				} else {
					gradients[i] = 0.0
				}
			}
			iterData.Targets = gradients
		} else {
			iterData.Targets = residuals
		}

		// Feature subsampling per tree
		treeFeatures := allFeatures
		if colsampleByTree > 0.0 && colsampleByTree < 1.0 {
			treeFeatures = sampleFeatures(allFeatures, colsampleByTree, ws.TreeFeatureIdxs, ws.PermBuf, rng)
		}

		treeNodes := trainLeafwiseTreeColumnarWithColsample(ws, iterData, params.MaxDepth, params.MaxLeaves, params.MinChildSamples, params.L2LeafReg, maxLeaf, params.LossFunction, params.Quantile, residuals, colsampleByNode, treeFeatures, rng)
		trees = append(trees, Tree{Nodes: treeNodes})

		for i := 0; i < n; i++ {
			preds[i] += params.LearningRate * EvaluateTreeColumnar(treeNodes, iterData, i)
		}

		if useEarlyStopping {
			for i := 0; i < valData.NumSamples; i++ {
				valPreds[i] += params.LearningRate * EvaluateTreeColumnar(treeNodes, valData, i)
			}
			valLoss := computeValidationLoss(valData, valPreds, isPinball, params.Quantile)
			if valLoss < bestValLoss-1e-12 {
				bestValLoss = valLoss
				bestIter = m
				patienceCount = 0
			} else {
				patienceCount++
				if patienceCount >= params.EarlyStoppingPatience {
					if bestIter >= 0 {
						trees = trees[:bestIter+1]
					}
					break
				}
			}
		}
	}

	return trees, baseValue
}

// TrainLeafwiseGBDT fits a leaf-wise (LightGBM-style) gradient boosted decision tree ensemble.
func TrainLeafwiseGBDT(X []FeatureVector, Y []float64, W []float64, params LightGBMParams) ([]Tree, float64) {
	if len(X) != len(Y) {
		panic(fmt.Sprintf("treeboost: length mismatch: len(X)=%d != len(Y)=%d", len(X), len(Y)))
	}
	if len(W) > 0 && len(W) != len(Y) {
		panic(fmt.Sprintf("treeboost: length mismatch: len(W)=%d != len(Y)=%d", len(W), len(Y)))
	}
	data := ToColumnar(X, Y, W)
	return TrainLeafwiseGBDTColumnar(data, params)
}

func trainExtraTreeNodes(ws *TrainWorkspace, rng *rand.Rand, data ColumnarDataset, sampleIdxs []int, start, end int, depth int, maxDepth int, minSamplesLeaf int, maxFeatures float64, maxLeaf float64, nodes *[]Node, lossFunction string, quantile float64) int {
	if minSamplesLeaf < 1 {
		minSamplesLeaf = 1
	}
	if maxDepth < 0 {
		maxDepth = 0
	}
	if maxFeatures <= 0.0 || math.IsNaN(maxFeatures) || math.IsInf(maxFeatures, 0) {
		maxFeatures = 1.0
	} else if maxFeatures > 1.0 {
		maxFeatures = 1.0
	}
	if data.NumSamples > 0 && len(data.Weights) == 0 {
		weights := make([]float64, data.NumSamples)
		for i := range weights {
			weights[i] = 1.0
		}
		data.Weights = weights
	}
	nSamples := end - start
	currIdx := len(*nodes)
	*nodes = append(*nodes, Node{NodeID: currIdx, SplitFeature: -1, LeftChild: -1, RightChild: -1})

	if nSamples < minSamplesLeaf || depth >= maxDepth {
		leafVal := 0.0
		if lossFunction == LossPinball {
			leafVal = weightedQuantileIndexed(sampleIdxs[start:end], data.Targets, data.Weights, quantile, &ws.QuantileScratch)
		} else {
			var sumW, sumWY float64
			for i := start; i < end; i++ {
				idx := sampleIdxs[i]
				w := data.Weights[idx]
				sumW += w
				sumWY += w * data.Targets[idx]
			}
			if sumW > 1e-9 {
				den := sumW
				if den < 1.0 {
					den = 1.0
				}
				leafVal = sumWY / den
			}
		}
		(*nodes)[currIdx].LeafValue = clampLeafValue(leafVal, maxLeaf)
		return currIdx
	}

	var sumW, sumWY, sumWY2 float64
	for i := start; i < end; i++ {
		idx := sampleIdxs[i]
		w := data.Weights[idx]
		y := data.Targets[idx]
		sumW += w
		sumWY += w * y
		sumWY2 += w * y * y
	}
	if sumW <= 1e-9 {
		(*nodes)[currIdx].LeafValue = 0.0
		return currIdx
	}
	var defaultLeafVal float64
	if lossFunction == LossPinball {
		defaultLeafVal = clampLeafValue(weightedQuantileIndexed(sampleIdxs[start:end], data.Targets, data.Weights, quantile, &ws.QuantileScratch), maxLeaf)
	} else {
		den := sumW
		if den < 1.0 {
			den = 1.0
		}
		defaultLeafVal = clampLeafValue(sumWY/den, maxLeaf)
	}
	errVal := sumWY2 - (sumWY*sumWY)/sumW
	if errVal <= 1e-9 || math.IsNaN(errVal) {
		(*nodes)[currIdx].LeafValue = defaultLeafVal
		return currIdx
	}

	numFeatures := data.NumFeatures
	featuresToTry := int(math.Ceil(float64(numFeatures) * maxFeatures))
	if featuresToTry < 1 {
		featuresToTry = 1
	}

	perm := ws.PermBuf[:numFeatures]
	for i := 0; i < numFeatures; i++ {
		perm[i] = i
	}
	for i := numFeatures - 1; i > 0; i-- {
		j := rng.Intn(i + 1)
		perm[i], perm[j] = perm[j], perm[i]
	}

	bestFeature := -1
	bestVal := 0.0
	bestScore := math.MaxFloat64

	featuresChecked := 0
	for _, f := range perm {
		if featuresChecked >= featuresToTry {
			break
		}

		fOffset := f * data.NumSamples
		firstIdx := sampleIdxs[start]
		minF := data.Features[fOffset+firstIdx]
		maxF := minF
		for i := start + 1; i < end; i++ {
			v := data.Features[fOffset+sampleIdxs[i]]
			if v < minF {
				minF = v
			}
			if v > maxF {
				maxF = v
			}
		}

		if minF == maxF {
			continue
		}

		featuresChecked++
		val := minF + rng.Float64()*(maxF-minF)
		if math.IsNaN(val) || math.IsInf(val, 0) {
			continue
		}

		var sumWLeft, sumWYLeft, sumWY2Left float64
		var sumWRight, sumWYRight, sumWY2Right float64
		var countLeft, countRight int

		for i := start; i < end; i++ {
			idx := sampleIdxs[i]
			w := data.Weights[idx]
			y := data.Targets[idx]
			if data.Features[fOffset+idx] < val {
				sumWLeft += w
				sumWYLeft += w * y
				sumWY2Left += w * y * y
				countLeft++
			} else {
				sumWRight += w
				sumWYRight += w * y
				sumWY2Right += w * y * y
				countRight++
			}
		}

		if countLeft == 0 || countRight == 0 {
			continue
		}

		var errLeft, errRight float64
		if sumWLeft > 1e-9 {
			errLeft = sumWY2Left - (sumWYLeft*sumWYLeft)/sumWLeft
			if errLeft < 0 || math.IsNaN(errLeft) || math.IsInf(errLeft, 0) {
				errLeft = 0
			}
		}
		if sumWRight > 1e-9 {
			errRight = sumWY2Right - (sumWYRight*sumWYRight)/sumWRight
			if errRight < 0 || math.IsNaN(errRight) || math.IsInf(errRight, 0) {
				errRight = 0
			}
		}

		score := errLeft + errRight
		if math.IsNaN(score) || math.IsInf(score, 0) {
			continue
		}
		if score < bestScore {
			bestScore = score
			bestFeature = f
			bestVal = val
		}
	}

	if bestFeature == -1 {
		(*nodes)[currIdx].LeafValue = defaultLeafVal
		return currIdx
	}

	mid := partitionSamplesInPlace(sampleIdxs, start, end, data.Features, bestFeature*data.NumSamples, bestVal)
	if mid == start || mid == end {
		(*nodes)[currIdx].LeafValue = defaultLeafVal
		return currIdx
	}

	(*nodes)[currIdx].SplitFeature = bestFeature
	(*nodes)[currIdx].SplitValue = bestVal

	leftIdx := trainExtraTreeNodes(ws, rng, data, sampleIdxs, start, mid, depth+1, maxDepth, minSamplesLeaf, maxFeatures, maxLeaf, nodes, lossFunction, quantile)
	rightIdx := trainExtraTreeNodes(ws, rng, data, sampleIdxs, mid, end, depth+1, maxDepth, minSamplesLeaf, maxFeatures, maxLeaf, nodes, lossFunction, quantile)

	(*nodes)[currIdx].LeftChild = leftIdx
	(*nodes)[currIdx].RightChild = rightIdx

	return currIdx
}

func trainExtraTree(ws *TrainWorkspace, rng *rand.Rand, X []FeatureVector, Y []float64, W []float64, indices []int, depth int, maxDepth int, minSamplesLeaf int, maxFeatures float64, maxLeaf float64) []Node {
	data := ToColumnar(X, Y, W)
	sampleIdxs := make([]int, len(indices))
	copy(sampleIdxs, indices)
	nodes := make([]Node, 0, 64)
	trainExtraTreeNodes(ws, rng, data, sampleIdxs, 0, len(indices), depth, maxDepth, minSamplesLeaf, maxFeatures, maxLeaf, &nodes, "", 0.0)
	return nodes
}

// TrainExtraTreesColumnar fits an ExtraTrees regression ensemble on ColumnarDataset without per-iteration conversions.
func TrainExtraTreesColumnar(data ColumnarDataset, params ExtraTreesParams) []Tree {
	var trees []Tree
	if params.LossFunction == LossPinball {
		validateQuantileParams(params.LossFunction, params.Quantile)
	}
	n := data.NumSamples
	if n == 0 || params.Estimators <= 0 {
		return trees
	}
	if len(data.Targets) != n {
		panic(fmt.Sprintf("treeboost: length mismatch: len(Targets)=%d != NumSamples=%d", len(data.Targets), n))
	}
	if len(data.Weights) > 0 && len(data.Weights) != n {
		panic(fmt.Sprintf("treeboost: length mismatch: len(Weights)=%d != NumSamples=%d", len(data.Weights), n))
	}
	if len(data.Weights) == 0 {
		weights := make([]float64, n)
		for i := range weights {
			weights[i] = 1.0
		}
		data.Weights = weights
	}
	maxLeaf := resolveMaxLeafValue(params.MaxLeafValue)
	trees = make([]Tree, 0, params.Estimators)
	rng := rand.New(rand.NewSource(42))

	ws := acquireWorkspace(data.NumSamples, data.NumFeatures)
	defer releaseWorkspace(ws)

	sampleIdxs := make([]int, data.NumSamples)
	minSamplesLeaf := params.MinSamplesLeaf
	if minSamplesLeaf < 1 {
		minSamplesLeaf = 1
	}

	for m := 0; m < params.Estimators; m++ {
		for i := range sampleIdxs {
			sampleIdxs[i] = i
		}

		nodes := make([]Node, 0, 64)
		trainExtraTreeNodes(ws, rng, data, sampleIdxs, 0, data.NumSamples, 0, 12, minSamplesLeaf, params.MaxFeatures, maxLeaf, &nodes, params.LossFunction, params.Quantile)
		trees = append(trees, Tree{Nodes: nodes})
	}

	return trees
}

// TrainExtraTrees fits an Extremely Randomized Trees (ExtraTrees) regression ensemble.
func TrainExtraTrees(X []FeatureVector, Y []float64, W []float64, params ExtraTreesParams) []Tree {
	if len(X) != len(Y) {
		panic(fmt.Sprintf("treeboost: length mismatch: len(X)=%d != len(Y)=%d", len(X), len(Y)))
	}
	if len(W) > 0 && len(W) != len(Y) {
		panic(fmt.Sprintf("treeboost: length mismatch: len(W)=%d != len(Y)=%d", len(W), len(Y)))
	}
	data := ToColumnar(X, Y, W)
	return TrainExtraTreesColumnar(data, params)
}

// ScaleTreeLeavesInPlace scales tree leaf values in place without allocating a new slice.
func ScaleTreeLeavesInPlace(tree *Tree, factor float64) {
	for i := range tree.Nodes {
		if tree.Nodes[i].SplitFeature == -1 {
			tree.Nodes[i].LeafValue *= factor
		}
	}
}

func TrainEnsembleWithWeights(X []FeatureVector, Y []float64, W []float64, config EnsembleConfig) *Model {
	validateEarlyStopping(config.EarlyStoppingPatience)
	validateEarlyStopping(config.CatBoost.EarlyStoppingPatience)
	validateEarlyStopping(config.LightGBM.EarlyStoppingPatience)
	if config.ColsampleByTree != 0.0 {
		validateColsample("colsample_bytree", config.ColsampleByTree)
	}
	if config.ColsampleByNode != 0.0 {
		validateColsample("colsample_bynode", config.ColsampleByNode)
	}
	if config.CatBoost.ColsampleByTree != 0.0 {
		validateColsample("colsample_bytree", config.CatBoost.ColsampleByTree)
	}
	if config.CatBoost.ColsampleByNode != 0.0 {
		validateColsample("colsample_bynode", config.CatBoost.ColsampleByNode)
	}
	if config.LightGBM.ColsampleByTree != 0.0 {
		validateColsample("colsample_bytree", config.LightGBM.ColsampleByTree)
	}
	if config.LightGBM.ColsampleByNode != 0.0 {
		validateColsample("colsample_bynode", config.LightGBM.ColsampleByNode)
	}

	if len(X) != len(Y) {
		panic(fmt.Sprintf("treeboost: length mismatch: len(X)=%d != len(Y)=%d", len(X), len(Y)))
	}
	if len(W) > 0 && len(W) != len(Y) {
		panic(fmt.Sprintf("treeboost: length mismatch: len(W)=%d != len(Y)=%d", len(W), len(Y)))
	}
	if len(W) == 0 && len(Y) > 0 {
		W = make([]float64, len(Y))
		for i := range W {
			W[i] = 1.0
		}
	}
	if config.LossFunction == LossPinball {
		validateQuantileParams(config.LossFunction, config.Quantile)
		if config.CatBoost.LossFunction == "" {
			config.CatBoost.LossFunction = LossPinball
			config.CatBoost.Quantile = config.Quantile
		}
		if config.LightGBM.LossFunction == "" {
			config.LightGBM.LossFunction = LossPinball
			config.LightGBM.Quantile = config.Quantile
		}
		if config.ExtraTrees.LossFunction == "" {
			config.ExtraTrees.LossFunction = LossPinball
			config.ExtraTrees.Quantile = config.Quantile
		}
	}
	cbParams := config.CatBoost
	if cbParams.LossFunction == LossPinball {
		validateQuantileParams(cbParams.LossFunction, cbParams.Quantile)
	}
	lgbParams := config.LightGBM
	if lgbParams.LossFunction == LossPinball {
		validateQuantileParams(lgbParams.LossFunction, lgbParams.Quantile)
	}
	etParams := config.ExtraTrees
	if etParams.LossFunction == LossPinball {
		validateQuantileParams(etParams.LossFunction, etParams.Quantile)
	}

	// Propagate EnsembleConfig level colsample and early stopping parameters if not explicitly set on sub-models
	if cbParams.ColsampleByTree <= 0.0 && config.ColsampleByTree > 0.0 {
		cbParams.ColsampleByTree = config.ColsampleByTree
	}
	if cbParams.ColsampleByNode <= 0.0 && config.ColsampleByNode > 0.0 {
		cbParams.ColsampleByNode = config.ColsampleByNode
	}
	if cbParams.EarlyStoppingPatience <= 0 && config.EarlyStoppingPatience > 0 {
		cbParams.EarlyStoppingPatience = config.EarlyStoppingPatience
	}
	if cbParams.ValidationData.NumSamples == 0 && config.ValidationData.NumSamples > 0 {
		cbParams.ValidationData = config.ValidationData
	}
	if len(cbParams.ValidationX) == 0 && len(config.ValidationX) > 0 {
		cbParams.ValidationX = config.ValidationX
		cbParams.ValidationY = config.ValidationY
		cbParams.ValidationW = config.ValidationW
	}

	if lgbParams.ColsampleByTree <= 0.0 && config.ColsampleByTree > 0.0 {
		lgbParams.ColsampleByTree = config.ColsampleByTree
	}
	if lgbParams.ColsampleByNode <= 0.0 && config.ColsampleByNode > 0.0 {
		lgbParams.ColsampleByNode = config.ColsampleByNode
	}
	if lgbParams.EarlyStoppingPatience <= 0 && config.EarlyStoppingPatience > 0 {
		lgbParams.EarlyStoppingPatience = config.EarlyStoppingPatience
	}
	if lgbParams.ValidationData.NumSamples == 0 && config.ValidationData.NumSamples > 0 {
		lgbParams.ValidationData = config.ValidationData
	}
	if len(lgbParams.ValidationX) == 0 && len(config.ValidationX) > 0 {
		lgbParams.ValidationX = config.ValidationX
		lgbParams.ValidationY = config.ValidationY
		lgbParams.ValidationW = config.ValidationW
	}

	data := ToColumnar(X, Y, W)

	maxLeaf := resolveMaxLeafValue(config.MaxLeafValue)
	if cbParams.MaxLeafValue <= 0.0 || cbParams.MaxLeafValue > maxLeaf || math.IsNaN(cbParams.MaxLeafValue) || math.IsInf(cbParams.MaxLeafValue, 0) {
		cbParams.MaxLeafValue = maxLeaf
	}
	if lgbParams.MaxLeafValue <= 0.0 || lgbParams.MaxLeafValue > maxLeaf || math.IsNaN(lgbParams.MaxLeafValue) || math.IsInf(lgbParams.MaxLeafValue, 0) {
		lgbParams.MaxLeafValue = maxLeaf
	}
	if etParams.MaxLeafValue <= 0.0 || etParams.MaxLeafValue > maxLeaf || math.IsNaN(etParams.MaxLeafValue) || math.IsInf(etParams.MaxLeafValue, 0) {
		etParams.MaxLeafValue = maxLeaf
	}

	cbTrees, cbBase := TrainSymmetricGBDTColumnar(data, cbParams)
	lgbTrees, lgbBase := TrainLeafwiseGBDTColumnar(data, lgbParams)
	etTrees := TrainExtraTreesColumnar(data, etParams)

	combinedBase := config.CatBoostWeight*cbBase + config.LightGBMWeight*lgbBase
	combinedBase = clampLeafValue(combinedBase, maxLeaf)

	totalTrees := len(cbTrees) + len(lgbTrees) + len(etTrees)
	combinedTrees := make([]Tree, 0, totalTrees)

	for i := range cbTrees {
		ScaleTreeLeavesInPlace(&cbTrees[i], config.CatBoostWeight*config.CatBoost.LearningRate)
		for j := range cbTrees[i].Nodes {
			if cbTrees[i].Nodes[j].SplitFeature == -1 {
				cbTrees[i].Nodes[j].LeafValue = clampLeafValue(cbTrees[i].Nodes[j].LeafValue, maxLeaf)
			}
		}
		combinedTrees = append(combinedTrees, cbTrees[i])
	}

	for i := range lgbTrees {
		ScaleTreeLeavesInPlace(&lgbTrees[i], config.LightGBMWeight*config.LightGBM.LearningRate)
		for j := range lgbTrees[i].Nodes {
			if lgbTrees[i].Nodes[j].SplitFeature == -1 {
				lgbTrees[i].Nodes[j].LeafValue = clampLeafValue(lgbTrees[i].Nodes[j].LeafValue, maxLeaf)
			}
		}
		combinedTrees = append(combinedTrees, lgbTrees[i])
	}

	etScale := 0.0
	if config.ExtraTrees.Estimators > 0 {
		etScale = config.ExtraTreesWeight / float64(config.ExtraTrees.Estimators)
	}
	for i := range etTrees {
		ScaleTreeLeavesInPlace(&etTrees[i], etScale)
		for j := range etTrees[i].Nodes {
			if etTrees[i].Nodes[j].SplitFeature == -1 {
				etTrees[i].Nodes[j].LeafValue = clampLeafValue(etTrees[i].Nodes[j].LeafValue, maxLeaf)
			}
		}
		combinedTrees = append(combinedTrees, etTrees[i])
	}

	model := &Model{
		TrainedAt:    time.Now().UTC(),
		BaseValue:    combinedBase,
		Trees:        combinedTrees,
		LearningRate: 1.0,
		LossFunction: config.LossFunction,
		Quantile:     config.Quantile,
	}
	model.ComputeMetrics(X, Y)
	return model
}

// TrainEnsemble trains a multi-algorithm tree ensemble combining CatBoost, LightGBM, and ExtraTrees models.
func TrainEnsemble(X []FeatureVector, Y []float64, config EnsembleConfig) *Model {
	return TrainEnsembleWithWeights(X, Y, ComputeSampleWeights(Y), config)
}

// TrainQuantileEnsembleWithWeights trains a multi-algorithm quantile regression ensemble
// for target quantile using explicit sample weights W.
func TrainQuantileEnsembleWithWeights(X []FeatureVector, Y []float64, W []float64, quantile float64, config EnsembleConfig) *Model {
	validateQuantileParams(LossPinball, quantile)
	if len(X) != len(Y) {
		panic(fmt.Sprintf("treeboost: length mismatch: len(X)=%d != len(Y)=%d", len(X), len(Y)))
	}
	if len(W) > 0 && len(W) != len(Y) {
		panic(fmt.Sprintf("treeboost: length mismatch: len(W)=%d != len(Y)=%d", len(W), len(Y)))
	}
	if len(W) == 0 && len(Y) > 0 {
		W = make([]float64, len(Y))
		for i := range W {
			W[i] = 1.0
		}
	}

	config.LossFunction = LossPinball
	config.Quantile = quantile
	config.CatBoost.LossFunction = LossPinball
	config.CatBoost.Quantile = quantile
	config.LightGBM.LossFunction = LossPinball
	config.LightGBM.Quantile = quantile
	config.ExtraTrees.LossFunction = LossPinball
	config.ExtraTrees.Quantile = quantile

	totalWeight := config.CatBoostWeight + config.LightGBMWeight
	if config.ExtraTrees.Estimators > 0 {
		totalWeight += config.ExtraTreesWeight
	} else {
		config.ExtraTreesWeight = 0.0
	}
	if totalWeight > 1e-9 && math.Abs(totalWeight-1.0) > 1e-9 {
		config.CatBoostWeight /= totalWeight
		config.LightGBMWeight /= totalWeight
		if config.ExtraTrees.Estimators > 0 {
			config.ExtraTreesWeight /= totalWeight
		}
	}

	return TrainEnsembleWithWeights(X, Y, W, config)
}

// TrainQuantileEnsemble trains a multi-algorithm quantile regression ensemble
// for target quantile defaulting to uniform sample weights (w_i = 1.0).
func TrainQuantileEnsemble(X []FeatureVector, Y []float64, quantile float64, config EnsembleConfig) *Model {
	return TrainQuantileEnsembleWithWeights(X, Y, nil, quantile, config)
}
