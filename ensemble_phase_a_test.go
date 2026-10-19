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
	"math"
	"math/rand"
	"reflect"
	"runtime"
	"sync"
	"testing"
)

func phaseAData(n, nFeatures int, seed int64) ([]FeatureVector, []float64, []float64) {
	rng := rand.New(rand.NewSource(seed))
	X := make([]FeatureVector, n)
	Y := make([]float64, n)
	W := make([]float64, n)
	for i := range X {
		v := make([]float64, nFeatures)
		for f := range v {
			v[f] = rng.NormFloat64()
		}
		X[i] = FeatureVector{Values: v}
		Y[i] = 2.0*v[0] - 1.5*v[1]*v[2] + 0.3*rng.NormFloat64()
		W[i] = 0.5 + rng.Float64()
	}
	return X, Y, W
}

// bruteForceLevelSplit evaluates every bin boundary for feature f across all non-empty nodes of the level
// using explicit per-node left/right sums, returning the best boundary index and its regularized score.
func bruteForceLevelSplit(data ColumnarDataset, f, levelStart, levelEnd int, ranges []nodeRange, sampleIdxs []int, lambda float64) (int, float64) {
	numBins := len(data.Binned.BinBoundaries[f])
	fOffset := f * data.NumSamples
	bestB, bestScore := -1, math.MaxFloat64
	for b := 0; b < numBins-1; b++ {
		var score float64
		splitsSomething := false
		for n := levelStart; n < levelEnd; n++ {
			r := ranges[n]
			var lw, lwy, rw, rwy float64
			var lc, rc int
			for i := r.start; i < r.end; i++ {
				idx := sampleIdxs[i]
				w, y := data.Weights[idx], data.Targets[idx]
				if int(data.Binned.Bins[fOffset+idx]) <= b {
					lw += w
					lwy += w * y
					lc++
				} else {
					rw += w
					rwy += w * y
					rc++
				}
			}
			if lc > 0 && rc > 0 {
				splitsSomething = true
			}
			if lw+lambda > 1e-9 {
				score -= lwy * lwy / (lw + lambda)
			}
			if rw+lambda > 1e-9 {
				score -= rwy * rwy / (rw + lambda)
			}
		}
		if splitsSomething && score < bestScore {
			bestB, bestScore = b, score
		}
	}
	return bestB, bestScore
}

func TestSymmetricLevelHistogram_MatchesBruteForcePerNode(t *testing.T) {
	X, Y, W := phaseAData(600, 6, 7)
	data := ToColumnar(X, Y, W)

	ws := acquireWorkspace(data.NumSamples, data.NumFeatures)
	defer releaseWorkspace(ws)

	// Build a depth-2 level: split the root on feature 0, then evaluate level 1 (two nodes).
	ws.Ranges = ws.Ranges[:0]
	for i := 0; i < 7; i++ {
		ws.Ranges = append(ws.Ranges, nodeRange{})
	}
	ws.Ranges[0] = nodeRange{start: 0, end: data.NumSamples}
	mid := partitionSamplesInPlace(ws.SampleIdxs, 0, data.NumSamples, data.Features, 0, 0.0)
	ws.Ranges[1] = nodeRange{start: 0, end: mid}
	ws.Ranges[2] = nodeRange{start: mid, end: data.NumSamples}

	for _, lambda := range []float64{0.0, 3.0, 50.0} {
		for f := 0; f < data.NumFeatures; f++ {
			got := evaluateSymmetricFeatureSplits(data, f, 1, 3, ws.Ranges, ws.SampleIdxs, lambda, ws)
			wantB, wantScore := bruteForceLevelSplit(data, f, 1, 3, ws.Ranges, ws.SampleIdxs, lambda)
			if wantB < 0 {
				if got.feature != -1 {
					t.Fatalf("f=%d lambda=%v: expected no split, got %+v", f, lambda, got)
				}
				continue
			}
			bounds := data.Binned.BinBoundaries[f]
			wantVal := bounds[wantB]*0.5 + bounds[wantB+1]*0.5
			if got.feature != f || got.val != wantVal {
				t.Fatalf("f=%d lambda=%v: got (feature=%d val=%v), want val=%v", f, lambda, got.feature, got.val, wantVal)
			}
			if math.Abs(got.score-wantScore) > 1e-6*math.Max(1, math.Abs(wantScore)) {
				t.Fatalf("f=%d lambda=%v: score %v != brute force %v", f, lambda, got.score, wantScore)
			}
		}
	}
}

func TestSymmetricLevelHistogram_DoesNotAllocate(t *testing.T) {
	X, Y, W := phaseAData(500, 4, 11)
	data := ToColumnar(X, Y, W)
	ws := acquireWorkspace(data.NumSamples, data.NumFeatures)
	defer releaseWorkspace(ws)
	ranges := []nodeRange{{start: 0, end: data.NumSamples}}

	allocs := testing.AllocsPerRun(20, func() {
		evaluateSymmetricFeatureSplits(data, 0, 0, 1, ranges, ws.SampleIdxs, 1.0, ws)
	})
	if allocs != 0 {
		t.Fatalf("binned symmetric split evaluation allocated %v times per run, want 0", allocs)
	}
}

func TestSymmetricTree_LevelHistogramFindsLevelOptimalThreshold(t *testing.T) {
	// y = 4*[x0>0] + s*[x1>-0.5] with s = +1 when x0>0 and -1 otherwise. The marginal effect of x1 cancels at
	// the root, so a root-only histogram cannot see the x1 threshold; the level-wise histograms across the two
	// depth-1 nodes recover it.
	const n = 1200
	rng := rand.New(rand.NewSource(3))
	X := make([]FeatureVector, n)
	Y := make([]float64, n)
	W := make([]float64, n)
	for i := range X {
		a, b := rng.Float64()*2-1, rng.Float64()*2-1
		X[i] = FeatureVector{Values: []float64{a, b}}
		W[i] = 1
		s := -1.0
		if a > 0 {
			Y[i] += 4
			s = 1.0
		}
		if b > -0.5 {
			Y[i] += s
		}
	}
	data := ToColumnar(X, Y, W)
	ws := acquireWorkspace(n, 2)
	defer releaseWorkspace(ws)

	nodes := trainSymmetricTreeColumnar(ws, data, 2, 0.0, 100.0, "", 0, nil)
	if len(nodes) < 7 || nodes[0].SplitFeature != 0 {
		t.Fatalf("expected root split on feature 0, got nodes %+v", nodes)
	}
	for _, idx := range []int{1, 2} {
		if nodes[idx].SplitFeature != 1 || math.Abs(nodes[idx].SplitValue-(-0.5)) > 0.1 {
			t.Fatalf("level-1 node %d split = (feature %d, value %v), want feature 1 near -0.5", idx, nodes[idx].SplitFeature, nodes[idx].SplitValue)
		}
	}
	var sse float64
	for i := 0; i < n; i++ {
		d := Y[i] - EvaluateTreeColumnar(nodes, data, i)
		sse += d * d
	}
	if mse := sse / n; mse > 0.1 {
		t.Fatalf("depth-2 symmetric tree MSE = %v, want < 0.1", mse)
	}
}

func TestIntraTreeParallelGate_Constant(t *testing.T) {
	if minSamplesForIntraTreeParallel != 4096 {
		t.Fatalf("minSamplesForIntraTreeParallel = %d, want 4096", minSamplesForIntraTreeParallel)
	}
}

func TestEnsemble_SmallNSequentialMatchesParallelGOMAXPROCS(t *testing.T) {
	X, Y, W := phaseAData(1500, 20, 21)
	cfg := DefaultQuantileEnsembleConfig(0.5)
	cfg.CatBoost.Iterations = 10
	cfg.LightGBM.Estimators = 10
	cfg.ExtraTrees.Estimators = 0
	cfg.ColsampleByTree = 0.8
	cfg.ColsampleByNode = 0.8

	prev := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prev)
	single := TrainQuantileEnsembleWithWeights(X, Y, W, 0.5, cfg)
	runtime.GOMAXPROCS(8)
	multi := TrainQuantileEnsembleWithWeights(X, Y, W, 0.5, cfg)

	if !reflect.DeepEqual(single.Trees, multi.Trees) || single.BaseValue != multi.BaseValue {
		t.Fatal("small-N training must be identical regardless of GOMAXPROCS")
	}
}

func modelsEqual(a, b *Model) bool {
	return a.BaseValue == b.BaseValue && reflect.DeepEqual(a.Trees, b.Trees) &&
		a.MAE == b.MAE && a.RMSE == b.RMSE && a.RSquared == b.RSquared && a.PinballLoss == b.PinballLoss
}

func TestTrainEnsembleColumnar_MatchesLegacyEntryPoint(t *testing.T) {
	X, Y, W := phaseAData(400, 8, 5)
	cfg := DefaultEnsembleConfig()
	cfg.CatBoost.Iterations = 8
	cfg.LightGBM.Estimators = 8
	cfg.ExtraTrees.Estimators = 5

	legacy := TrainEnsembleWithWeights(X, Y, W, cfg)
	prebinned := TrainEnsembleColumnar(ToColumnar(X, Y, W), cfg)
	if !modelsEqual(legacy, prebinned) {
		t.Fatal("TrainEnsembleColumnar(pre-binned) differs from TrainEnsembleWithWeights")
	}

	unbinned := ToColumnar(X, Y, W)
	unbinned.Binned = BinnedDataset{}
	fromUnbinned := TrainEnsembleColumnar(unbinned, cfg)
	if !modelsEqual(legacy, fromUnbinned) {
		t.Fatal("TrainEnsembleColumnar must quantize unbinned data identically to ToColumnar")
	}
	if len(unbinned.Binned.Bins) != 0 {
		t.Fatal("caller's ColumnarDataset must not be mutated")
	}

	// Metrics computed from the columnar layout must match row-oriented ComputeMetrics.
	check := *legacy
	check.MAE, check.RMSE, check.RSquared, check.PinballLoss = 0, 0, 0, 0
	check.ComputeMetrics(X, Y)
	if math.Abs(check.MAE-legacy.MAE) > 1e-12 || math.Abs(check.RMSE-legacy.RMSE) > 1e-12 || math.Abs(check.RSquared-legacy.RSquared) > 1e-12 {
		t.Fatalf("columnar metrics (%v,%v,%v) != ComputeMetrics (%v,%v,%v)", legacy.MAE, legacy.RMSE, legacy.RSquared, check.MAE, check.RMSE, check.RSquared)
	}
}

func TestTrainEnsembleColumnar_ReusesPreBinnedMatrix(t *testing.T) {
	X, Y, W := phaseAData(300, 5, 9)
	data := ToColumnar(X, Y, W)

	// Replace the bins with a coarse custom quantization; if training re-binned, results would match the default model.
	coarse := QuantizeDataset(data, 8)
	custom := data
	custom.Binned = coarse

	cfg := DefaultEnsembleConfig()
	cfg.CatBoost.Iterations = 5
	cfg.LightGBM.Estimators = 5
	cfg.ExtraTrees.Estimators = 0

	binsBefore := append([]uint8(nil), coarse.Bins...)
	reused := TrainEnsembleColumnar(custom, cfg)
	if !reflect.DeepEqual(binsBefore, coarse.Bins) {
		t.Fatal("pre-binned matrix must be treated as read-only")
	}
	defaultModel := TrainEnsembleColumnar(data, cfg)
	if reflect.DeepEqual(reused.Trees, defaultModel.Trees) {
		t.Fatal("coarse pre-binned input produced identical trees to 256-bin input; pre-binned matrix was not reused")
	}
}

func TestTrainEnsembleColumnar_InvalidShapesPanic(t *testing.T) {
	X, Y, W := phaseAData(50, 3, 1)
	cfg := DefaultEnsembleConfig()

	mustPanic := func(name string, fn func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Errorf("%s: expected panic", name)
			}
		}()
		fn()
	}

	data := ToColumnar(X, Y, W)
	badTargets := data
	badTargets.Targets = Y[:10]
	mustPanic("targets", func() { TrainEnsembleColumnar(badTargets, cfg) })

	badWeights := data
	badWeights.Weights = W[:10]
	mustPanic("weights", func() { TrainEnsembleColumnar(badWeights, cfg) })

	badBins := data
	badBins.Binned = QuantizeDataset(ToColumnar(X[:20], Y[:20], W[:20]), 256)
	mustPanic("binned shape", func() { TrainEnsembleColumnar(badBins, cfg) })

	mustPanic("quantile", func() { TrainQuantileEnsembleColumnar(data, 1.5, cfg) })
}

func TestTrainEnsembleColumnar_EmptyDataset(t *testing.T) {
	m := TrainEnsembleColumnar(ColumnarDataset{}, DefaultEnsembleConfig())
	if m == nil {
		t.Fatal("expected non-nil model for empty dataset")
	}
}

func TestTrainQuantileEnsembleColumnar_SharedDatasetConcurrent(t *testing.T) {
	X, Y, W := phaseAData(500, 10, 13)
	shared := ToColumnar(X, Y, W)
	cfg := DefaultEnsembleConfig()
	cfg.CatBoost.Iterations = 6
	cfg.LightGBM.Estimators = 6
	cfg.ExtraTrees.Estimators = 3

	quantiles := []float64{0.1, 0.5, 0.9}
	results := make([]*Model, len(quantiles))
	var wg sync.WaitGroup
	for i, q := range quantiles {
		wg.Add(1)
		go func(i int, q float64) {
			defer wg.Done()
			results[i] = TrainQuantileEnsembleColumnar(shared, q, cfg)
		}(i, q)
	}
	wg.Wait()

	for i, q := range quantiles {
		want := TrainQuantileEnsembleWithWeights(X, Y, W, q, cfg)
		if !modelsEqual(want, results[i]) {
			t.Fatalf("quantile %v: shared-dataset columnar model differs from legacy entry point", q)
		}
	}
}
