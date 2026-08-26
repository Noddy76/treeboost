package treeboost

import (
	"math"
	"math/rand"
	"sync"
	"time"
)

// CatBoostParams holds hyperparameters for Symmetric GBDT (obliviated trees).
// Reference: https://en.wikipedia.org/wiki/CatBoost
type CatBoostParams struct {
	Depth        int     `json:"depth"`
	Iterations   int     `json:"iterations"`
	LearningRate float64 `json:"learning_rate"`
	L2LeafReg    float64 `json:"l2_leaf_reg"`
}

// LightGBMParams holds hyperparameters for Leaf-wise GBDT (best-first tree growth).
// Reference: https://en.wikipedia.org/wiki/LightGBM
type LightGBMParams struct {
	MaxDepth        int     `json:"max_depth"`
	MaxLeaves       int     `json:"max_leaves"`
	MinChildSamples int     `json:"min_child_samples"`
	Estimators      int     `json:"estimators"`
	LearningRate    float64 `json:"learning_rate"`
}

// ExtraTreesParams holds hyperparameters for Extremely Randomized Trees.
// Reference: https://en.wikipedia.org/wiki/Random_forest#ExtraTrees
type ExtraTreesParams struct {
	Estimators     int     `json:"estimators"`
	MinSamplesLeaf int     `json:"min_samples_leaf"`
	MaxFeatures    float64 `json:"max_features"`
}

// EnsembleConfig specifies individual algorithm parameters and weighted contribution ratios.
type EnsembleConfig struct {
	CatBoostWeight   float64          `json:"catboost_weight"`
	LightGBMWeight   float64          `json:"lightgbm_weight"`
	ExtraTreesWeight float64          `json:"extratrees_weight"`
	CatBoost         CatBoostParams   `json:"catboost"`
	LightGBM         LightGBMParams   `json:"lightgbm"`
	ExtraTrees       ExtraTreesParams `json:"extratrees"`
}

// DefaultEnsembleConfig returns a balanced, standard default configuration combining
// Symmetric GBDT, Leaf-wise GBDT, and ExtraTrees into an equal-weight ensemble.
func DefaultEnsembleConfig() EnsembleConfig {
	return EnsembleConfig{
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
			MinChildSamples: 5,
			Estimators:      500,
			LearningRate:    0.05,
		},
		ExtraTrees: ExtraTreesParams{
			Estimators:     700,
			MinSamplesLeaf: 4,
			MaxFeatures:    1.0,
		},
	}
}

// ComputeSampleWeights calculates weights for observations using a log-distance from the target mean,
// emphasizing and penalizing extreme spikes/troughs.
func ComputeSampleWeights(Y []float64) []float64 {
	if len(Y) == 0 {
		return nil
	}
	var sum float64
	for _, val := range Y {
		sum += val
	}
	mean := sum / float64(len(Y))

	weights := make([]float64, len(Y))
	for i, y := range Y {
		diff := math.Abs(y - mean)
		w := 5.0*math.Log10(diff+10.0) - 4.0
		weights[i] = math.Round(w)
		if weights[i] < 1.0 {
			weights[i] = 1.0
		}
	}
	return weights
}

type nodeRange struct {
	start int
	end   int
}

// TrainWorkspace holds reusable slice buffers to eliminate heap allocations during tree construction.
type TrainWorkspace struct {
	SampleIdxs []int
	PermBuf    []int
	SplitBuf   []float64
	Ranges     []nodeRange
	Hist       FeatureHistogram
}

var trainWorkspacePool = sync.Pool{
	New: func() interface{} {
		return &TrainWorkspace{
			SampleIdxs: make([]int, 0, 4096),
			PermBuf:    make([]int, 0, 256),
			SplitBuf:   make([]float64, 50),
			Ranges:     make([]nodeRange, 0, 256),
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
	return ws
}

func releaseWorkspace(ws *TrainWorkspace) {
	if ws == nil {
		return
	}
	trainWorkspacePool.Put(ws)
}

// partitionSamplesInPlace partitions sampleIdxs[start:end] in-place around splitVal for feature f.
// Elements with feature < splitVal are placed in sampleIdxs[start:mid],
// and elements with feature >= splitVal in sampleIdxs[mid:end].
// Returns mid.
func partitionSamplesInPlace(sampleIdxs []int, start, end int, featureValues []float64, fOffset int, splitVal float64) int {
	i := start
	j := end - 1
	for i <= j {
		for i <= j && featureValues[fOffset+sampleIdxs[i]] < splitVal {
			i++
		}
		for i <= j && featureValues[fOffset+sampleIdxs[j]] >= splitVal {
			j--
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

func trainSymmetricTreeColumnar(ws *TrainWorkspace, data ColumnarDataset, depth int, lambda float64) []Node {
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

	for d := 0; d < depth; d++ {
		levelStart := (1 << d) - 1
		levelEnd := (1 << (d + 1)) - 1

		bestFeature := -1
		bestVal := 0.0
		bestScore := math.MaxFloat64

		for f := 0; f < data.NumFeatures; f++ {
			fOffset := f * data.NumSamples

			var candidates []float64
			if len(data.Binned.Bins) > 0 {
				BuildHistogramSlice(data.Binned.Bins, fOffset, data.Weights, data.Targets, ws.SampleIdxs, 0, data.NumSamples, &ws.Hist)
				totalW, totalWY := computeWeightedSumContiguous(data.Weights, data.Targets)
				val, _, ok := FindBestHistogramSplit(&ws.Hist, data.Binned.BinBoundaries[f], totalW, totalWY)
				if ok {
					candidates = []float64{val}
				}
			} else {
				candidates = getSplitCandidatesColumnar(data, ws.SampleIdxs[:data.NumSamples], f, ws)
			}
			if len(candidates) == 0 {
				continue
			}

			for _, val := range candidates {
				var totalScore float64
				for nIdx := levelStart; nIdx < levelEnd; nIdx++ {
					r := ws.Ranges[nIdx]
					if r.end-r.start == 0 {
						continue
					}
					var sumWLeft, sumWYLeft, sumWY2Left float64
					var sumWRight, sumWYRight, sumWY2Right float64
					for i := r.start; i < r.end; i++ {
						idx := ws.SampleIdxs[i]
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
						if errLeft < 0 {
							errLeft = 0
						}
					}
					if sumWRight > 1e-9 {
						errRight = sumWY2Right - (sumWYRight*sumWYRight)/sumWRight
						if errRight < 0 {
							errRight = 0
						}
					}
					totalScore += errLeft + errRight
				}

				if totalScore < bestScore {
					bestScore = totalScore
					bestFeature = f
					bestVal = val
				}
			}
		}

		if bestFeature == -1 {
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

	leafStart := (1 << depth) - 1
	leafEnd := (1 << (depth + 1)) - 1
	for nIdx := leafStart; nIdx < leafEnd; nIdx++ {
		r := ws.Ranges[nIdx]
		nodes[nIdx].SplitFeature = -1
		if r.end-r.start == 0 {
			nodes[nIdx].LeafValue = 0.0
			continue
		}
		var sumWY, sumW float64
		for i := r.start; i < r.end; i++ {
			idx := ws.SampleIdxs[i]
			sumWY += data.Weights[idx] * data.Targets[idx]
			sumW += data.Weights[idx]
		}
		nodes[nIdx].LeafValue = sumWY / (sumW + lambda)
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
		leftNode := nodes[node.LeftChild]
		rightNode := nodes[node.RightChild]
		if leftNode.SplitFeature == -1 && rightNode.SplitFeature == -1 && leftNode.LeafValue == 0.0 && rightNode.LeafValue == 0.0 {
			node.SplitFeature = -1
			node.LeafValue = 0.0
		}
	}
	prune(0)

	return nodes
}

func trainSymmetricTree(ws *TrainWorkspace, X []FeatureVector, Y []float64, W []float64, depth int, lambda float64) []Node {
	data := ToColumnar(X, Y, W)
	return trainSymmetricTreeColumnar(ws, data, depth, lambda)
}

// TrainSymmetricGBDTColumnar fits a CatBoost-style GBDT on ColumnarDataset without per-iteration conversions.
func TrainSymmetricGBDTColumnar(data ColumnarDataset, params CatBoostParams) ([]Tree, float64) {
	n := data.NumSamples
	if n == 0 {
		return nil, 0.0
	}
	sumW, sumWY := computeWeightedSumContiguous(data.Weights, data.Targets)
	baseValue := 0.0
	if sumW > 1e-9 {
		baseValue = sumWY / sumW
	}

	preds := make([]float64, n)
	for i := 0; i < n; i++ {
		preds[i] = baseValue
	}

	trees := make([]Tree, 0, params.Iterations)
	residuals := make([]float64, n)

	ws := acquireWorkspace(data.NumSamples, data.NumFeatures)
	defer releaseWorkspace(ws)

	iterData := data
	for m := 0; m < params.Iterations; m++ {
		for i := 0; i < n; i++ {
			residuals[i] = data.Targets[i] - preds[i]
		}
		iterData.Targets = residuals

		treeNodes := trainSymmetricTreeColumnar(ws, iterData, params.Depth, params.L2LeafReg)
		trees = append(trees, Tree{Nodes: treeNodes})

		for i := 0; i < n; i++ {
			preds[i] += params.LearningRate * EvaluateTreeColumnar(treeNodes, iterData, i)
		}
	}

	return trees, baseValue
}

// TrainSymmetricGBDT fits a symmetric (CatBoost-style) gradient boosted decision tree ensemble.
func TrainSymmetricGBDT(X []FeatureVector, Y []float64, W []float64, params CatBoostParams) ([]Tree, float64) {
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

func trainLeafwiseTreeColumnar(ws *TrainWorkspace, data ColumnarDataset, maxDepth int, maxLeaves int, minChildSamples int) []Node {
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
	if sumW > 1e-9 {
		meanRoot = sumWY / sumW
	}

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

	for len(activeLeaves) < maxLeaves {
		bestActiveIdx := -1
		bestGain := -1.0
		bestF := -1
		bestVal := 0.0
		var bestLeftMean, bestRightMean float64
		var bestLeftW, bestLeftWY, bestRightW, bestRightWY float64

		for i, nodeIdx := range activeLeaves {
			leaf := &buildNodes[nodeIdx]
			nSamples := leaf.end - leaf.start
			if leaf.depth >= maxDepth || nSamples < minChildSamples*2 {
				continue
			}

			for f := 0; f < data.NumFeatures; f++ {
				fOffset := f * data.NumSamples
				if len(data.Binned.Bins) > 0 {
					BuildHistogramSlice(data.Binned.Bins, fOffset, data.Weights, data.Targets, ws.SampleIdxs, leaf.start, leaf.end, &ws.Hist)
					val, _, leftW, leftWY, rightW, rightWY, ok := FindBestHistogramSplitWithCounts(
						&ws.Hist, data.Binned.BinBoundaries[f], leaf.sumW, leaf.sumWY, nSamples, minChildSamples)
					if !ok {
						continue
					}

					gain := (leftWY*leftWY/leftW + rightWY*rightWY/rightW) - (leaf.sumWY*leaf.sumWY/leaf.sumW)

					if gain > bestGain {
						bestGain = gain
						bestActiveIdx = i
						bestF = f
						bestVal = val
						bestLeftMean = leftWY / leftW
						bestRightMean = rightWY / rightW
						bestLeftW = leftW
						bestLeftWY = leftWY
						bestRightW = rightW
						bestRightWY = rightWY
					}
				} else {
					candidates := getSplitCandidatesColumnar(data, ws.SampleIdxs[leaf.start:leaf.end], f, ws)
					if len(candidates) == 0 {
						continue
					}

					for _, val := range candidates {
						var sumWLeft, sumWYLeft, sumWY2Left float64
						var sumWRight, sumWYRight, sumWY2Right float64
						var countLeft, countRight int
						for s := leaf.start; s < leaf.end; s++ {
							idx := ws.SampleIdxs[s]
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

						gain := (sumWYLeft*sumWYLeft/sumWLeft + sumWYRight*sumWYRight/sumWRight) - (leaf.sumWY*leaf.sumWY/leaf.sumW)

						if gain > bestGain {
							bestGain = gain
							bestActiveIdx = i
							bestF = f
							bestVal = val
							bestLeftMean = sumWYLeft / sumWLeft
							bestRightMean = sumWYRight / sumWRight
							bestLeftW = sumWLeft
							bestLeftWY = sumWYLeft
							bestRightW = sumWRight
							bestRightWY = sumWYRight
						}
					}
				}
			}
		}

		if bestActiveIdx == -1 || bestGain <= 1e-9 {
			break
		}

		parentIdx := activeLeaves[bestActiveIdx]
		parent := &buildNodes[parentIdx]
		parent.splitFeature = bestF
		parent.splitValue = bestVal

		mid := partitionSamplesInPlace(ws.SampleIdxs, parent.start, parent.end, data.Features, bestF*data.NumSamples, bestVal)
		if mid == parent.start || mid == parent.end {
			break
		}

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
		nodes[i] = Node{
			NodeID:       bn.nodeID,
			SplitFeature: bn.splitFeature,
			SplitValue:   bn.splitValue,
			LeafValue:    bn.leafValue,
			LeftChild:    bn.leftChild,
			RightChild:   bn.rightChild,
		}
	}

	return nodes
}

func trainLeafwiseTree(ws *TrainWorkspace, X []FeatureVector, Y []float64, W []float64, maxDepth int, maxLeaves int, minChildSamples int) []Node {
	data := ToColumnar(X, Y, W)
	return trainLeafwiseTreeColumnar(ws, data, maxDepth, maxLeaves, minChildSamples)
}

// TrainLeafwiseGBDTColumnar fits a leaf-wise GBDT on ColumnarDataset without per-iteration conversions.
func TrainLeafwiseGBDTColumnar(data ColumnarDataset, params LightGBMParams) ([]Tree, float64) {
	n := data.NumSamples
	if n == 0 {
		return nil, 0.0
	}
	sumW, sumWY := computeWeightedSumContiguous(data.Weights, data.Targets)
	baseValue := 0.0
	if sumW > 1e-9 {
		baseValue = sumWY / sumW
	}

	preds := make([]float64, n)
	for i := 0; i < n; i++ {
		preds[i] = baseValue
	}

	trees := make([]Tree, 0, params.Estimators)
	residuals := make([]float64, n)

	ws := acquireWorkspace(data.NumSamples, data.NumFeatures)
	defer releaseWorkspace(ws)

	iterData := data
	for m := 0; m < params.Estimators; m++ {
		for i := 0; i < n; i++ {
			residuals[i] = data.Targets[i] - preds[i]
		}
		iterData.Targets = residuals

		treeNodes := trainLeafwiseTreeColumnar(ws, iterData, params.MaxDepth, params.MaxLeaves, params.MinChildSamples)
		trees = append(trees, Tree{Nodes: treeNodes})

		for i := 0; i < n; i++ {
			preds[i] += params.LearningRate * EvaluateTreeColumnar(treeNodes, iterData, i)
		}
	}

	return trees, baseValue
}

// TrainLeafwiseGBDT fits a leaf-wise (LightGBM-style) gradient boosted decision tree ensemble.
func TrainLeafwiseGBDT(X []FeatureVector, Y []float64, W []float64, params LightGBMParams) ([]Tree, float64) {
	data := ToColumnar(X, Y, W)
	return TrainLeafwiseGBDTColumnar(data, params)
}

func trainExtraTreeNodes(ws *TrainWorkspace, rng *rand.Rand, data ColumnarDataset, sampleIdxs []int, start, end int, depth int, maxDepth int, minSamplesLeaf int, maxFeatures float64, nodes *[]Node) int {
	nSamples := end - start
	currIdx := len(*nodes)
	*nodes = append(*nodes, Node{NodeID: currIdx, SplitFeature: -1, LeftChild: -1, RightChild: -1})

	if nSamples < minSamplesLeaf || depth >= maxDepth {
		var sumW, sumWY float64
		for i := start; i < end; i++ {
			idx := sampleIdxs[i]
			w := data.Weights[idx]
			sumW += w
			sumWY += w * data.Targets[idx]
		}
		leafVal := 0.0
		if sumW > 1e-9 {
			leafVal = sumWY / sumW
		}
		(*nodes)[currIdx].LeafValue = leafVal
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
	meanVal := sumWY / sumW
	errVal := sumWY2 - (sumWY*sumWY)/sumW
	if errVal <= 1e-9 {
		(*nodes)[currIdx].LeafValue = meanVal
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
			if errLeft < 0 {
				errLeft = 0
			}
		}
		if sumWRight > 1e-9 {
			errRight = sumWY2Right - (sumWYRight*sumWYRight)/sumWRight
			if errRight < 0 {
				errRight = 0
			}
		}

		score := errLeft + errRight
		if score < bestScore {
			bestScore = score
			bestFeature = f
			bestVal = val
		}
	}

	if bestFeature == -1 {
		(*nodes)[currIdx].LeafValue = meanVal
		return currIdx
	}

	mid := partitionSamplesInPlace(sampleIdxs, start, end, data.Features, bestFeature*data.NumSamples, bestVal)
	if mid == start || mid == end {
		(*nodes)[currIdx].LeafValue = meanVal
		return currIdx
	}

	(*nodes)[currIdx].SplitFeature = bestFeature
	(*nodes)[currIdx].SplitValue = bestVal

	leftIdx := trainExtraTreeNodes(ws, rng, data, sampleIdxs, start, mid, depth+1, maxDepth, minSamplesLeaf, maxFeatures, nodes)
	rightIdx := trainExtraTreeNodes(ws, rng, data, sampleIdxs, mid, end, depth+1, maxDepth, minSamplesLeaf, maxFeatures, nodes)

	(*nodes)[currIdx].LeftChild = leftIdx
	(*nodes)[currIdx].RightChild = rightIdx

	return currIdx
}

func trainExtraTree(ws *TrainWorkspace, rng *rand.Rand, X []FeatureVector, Y []float64, W []float64, indices []int, depth int, maxDepth int, minSamplesLeaf int, maxFeatures float64) []Node {
	data := ToColumnar(X, Y, W)
	sampleIdxs := make([]int, len(indices))
	copy(sampleIdxs, indices)
	nodes := make([]Node, 0, 64)
	trainExtraTreeNodes(ws, rng, data, sampleIdxs, 0, len(indices), depth, maxDepth, minSamplesLeaf, maxFeatures, &nodes)
	return nodes
}

// TrainExtraTreesColumnar fits an ExtraTrees regression ensemble on ColumnarDataset without per-iteration conversions.
func TrainExtraTreesColumnar(data ColumnarDataset, params ExtraTreesParams) []Tree {
	var trees []Tree
	if data.NumSamples == 0 || params.Estimators <= 0 {
		return trees
	}
	trees = make([]Tree, 0, params.Estimators)
	rng := rand.New(rand.NewSource(42))

	ws := acquireWorkspace(data.NumSamples, data.NumFeatures)
	defer releaseWorkspace(ws)

	sampleIdxs := make([]int, data.NumSamples)

	for m := 0; m < params.Estimators; m++ {
		for i := range sampleIdxs {
			sampleIdxs[i] = i
		}

		nodes := make([]Node, 0, 64)
		trainExtraTreeNodes(ws, rng, data, sampleIdxs, 0, data.NumSamples, 0, 12, params.MinSamplesLeaf, params.MaxFeatures, &nodes)
		trees = append(trees, Tree{Nodes: nodes})
	}

	return trees
}

// TrainExtraTrees fits an Extremely Randomized Trees (ExtraTrees) regression ensemble.
func TrainExtraTrees(X []FeatureVector, Y []float64, W []float64, params ExtraTreesParams) []Tree {
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

// TrainEnsemble trains a multi-algorithm tree ensemble combining CatBoost, LightGBM, and ExtraTrees models.
func TrainEnsemble(X []FeatureVector, Y []float64, config EnsembleConfig) *Model {
	W := ComputeSampleWeights(Y)
	data := ToColumnar(X, Y, W)

	cbTrees, cbBase := TrainSymmetricGBDTColumnar(data, config.CatBoost)
	lgbTrees, lgbBase := TrainLeafwiseGBDTColumnar(data, config.LightGBM)
	etTrees := TrainExtraTreesColumnar(data, config.ExtraTrees)

	combinedBase := config.CatBoostWeight*cbBase + config.LightGBMWeight*lgbBase

	totalTrees := len(cbTrees) + len(lgbTrees) + len(etTrees)
	combinedTrees := make([]Tree, 0, totalTrees)

	for i := range cbTrees {
		ScaleTreeLeavesInPlace(&cbTrees[i], config.CatBoostWeight*config.CatBoost.LearningRate)
		combinedTrees = append(combinedTrees, cbTrees[i])
	}

	for i := range lgbTrees {
		ScaleTreeLeavesInPlace(&lgbTrees[i], config.LightGBMWeight*config.LightGBM.LearningRate)
		combinedTrees = append(combinedTrees, lgbTrees[i])
	}

	etScale := 0.0
	if config.ExtraTrees.Estimators > 0 {
		etScale = config.ExtraTreesWeight / float64(config.ExtraTrees.Estimators)
	}
	for i := range etTrees {
		ScaleTreeLeavesInPlace(&etTrees[i], etScale)
		combinedTrees = append(combinedTrees, etTrees[i])
	}

	model := &Model{
		TrainedAt:    time.Now().UTC(),
		BaseValue:    combinedBase,
		Trees:        combinedTrees,
		LearningRate: 1.0,
	}
	model.ComputeMetrics(X, Y)
	return model
}
