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
	"testing"
)

func TestLightGBM_L2LeafReg_DivergencePrevention(t *testing.T) {
	// Construct a synthetic dataset (N = 500) with extreme outlier weights on negative plunges
	const nSamples = 500
	const nFeatures = 5

	rng := rand.New(rand.NewSource(42))
	X := make([]FeatureVector, nSamples)
	Y := make([]float64, nSamples)
	W := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		vals := make([]float64, nFeatures)
		for f := 0; f < nFeatures; f++ {
			vals[f] = rng.Float64()*10.0 - 5.0
		}
		X[i] = FeatureVector{Values: vals}

		// Periodic plunges with negative targets
		if i%15 == 0 {
			Y[i] = -50.0 - rng.Float64()*50.0 // extreme negative plunge
			W[i] = 10.0                       // 10x penalty weight
		} else {
			Y[i] = 20.0 + vals[0]*2.0 - vals[1]*1.5 + rng.NormFloat64()
			W[i] = 1.0
		}
	}

	// 1. Train with L2LeafReg = 1.0 (regularized)
	regParams := LightGBMParams{
		MaxDepth:        5,
		MaxLeaves:       31,
		MinChildSamples: 20,
		Estimators:      300,
		LearningRate:    0.05,
		L2LeafReg:       1.0,
	}
	regTrees, regBase := TrainLeafwiseGBDT(X, Y, W, regParams)

	// Assert that with L2LeafReg > 0, all leaf values remain strictly bounded
	for tIdx, tree := range regTrees {
		for _, node := range tree.Nodes {
			if node.SplitFeature == -1 {
				if math.IsNaN(node.LeafValue) || math.IsInf(node.LeafValue, 0) {
					t.Fatalf("tree %d node %d has non-finite leaf value: %f", tIdx, node.NodeID, node.LeafValue)
				}
				if math.Abs(node.LeafValue) > 1000.0 {
					t.Errorf("tree %d node %d leaf value exploded: %f", tIdx, node.NodeID, node.LeafValue)
				}
			}
		}
	}

	// Assert that predictions remain strictly bounded (|pred| < 1000.0)
	for i, fv := range X {
		pred := regBase
		for _, tree := range regTrees {
			pred += regParams.LearningRate * EvaluateTree(tree.Nodes, fv)
		}
		if math.IsNaN(pred) || math.IsInf(pred, 0) {
			t.Fatalf("regularized prediction %d is non-finite: %f", i, pred)
		}
		if math.Abs(pred) > 1000.0 {
			t.Errorf("regularized prediction %d exploded: %f", i, pred)
		}
	}

	// Also test with full ensemble config (DefaultEnsembleConfig)
	ensembleCfg := DefaultEnsembleConfig()
	ensembleCfg.CatBoost.Iterations = 50
	ensembleCfg.LightGBM.Estimators = 50
	ensembleCfg.ExtraTrees.Estimators = 50
	ensembleModel := TrainEnsembleWithWeights(X, Y, W, ensembleCfg)
	for i, fv := range X {
		pred := ensembleModel.Predict(fv.Values)
		if math.IsNaN(pred) || math.IsInf(pred, 0) {
			t.Fatalf("ensemble prediction %d is non-finite: %f", i, pred)
		}
		if math.Abs(pred) > 1000.0 {
			t.Errorf("ensemble prediction %d exploded: %f", i, pred)
		}
	}
}

func TestLightGBM_BackwardCompatibility_L2LeafRegZero(t *testing.T) {
	// Verify that setting L2LeafReg: 0.0 runs without error and without div-by-zero
	X := []FeatureVector{
		{Values: []float64{1.0, 2.0}},
		{Values: []float64{2.0, 3.0}},
		{Values: []float64{3.0, 4.0}},
		{Values: []float64{4.0, 5.0}},
	}
	Y := []float64{10.0, 20.0, 30.0, 40.0}
	W := []float64{1.0, 1.0, 1.0, 1.0}

	params := LightGBMParams{
		MaxDepth:        2,
		MaxLeaves:       4,
		MinChildSamples: 1,
		Estimators:      5,
		LearningRate:    0.1,
		L2LeafReg:       0.0,
	}

	trees, base := TrainLeafwiseGBDT(X, Y, W, params)
	if len(trees) != 5 {
		t.Fatalf("expected 5 trees, got %d", len(trees))
	}
	if base <= 0 {
		t.Errorf("expected positive base value, got %f", base)
	}
}

func TestLightGBM_DiscreteFeatures_NoDivergence(t *testing.T) {
	const nTrain = 200
	const nHoldout = 50
	const nTotal = nTrain + nHoldout

	rng := rand.New(rand.NewSource(12345))

	XAll := make([]FeatureVector, nTotal)
	YAll := make([]float64, nTotal)
	WAll := make([]float64, nTotal)

	for i := 0; i < nTotal; i++ {
		feat0 := float64(rng.Intn(2))                       // Binary flag [0.0, 1.0]
		feat1 := float64(rng.Intn(7))                       // Day of week [0.0 ... 6.0]
		feat2 := math.Cos(float64(i) * 2.0 * math.Pi / 7.0) // Bounded cosine [-1.0 ... 1.0]

		XAll[i] = FeatureVector{Values: []float64{feat0, feat1, feat2}}
		// Target: continuous value correlated with day of week
		YAll[i] = 15.0 + 3.0*feat1 - 4.0*feat0 + 2.0*feat2 + rng.NormFloat64()*0.2
		WAll[i] = 1.0
	}

	XTrain := XAll[:nTrain]
	YTrain := YAll[:nTrain]
	WTrain := WAll[:nTrain]

	XHoldout := XAll[nTrain:]

	params := LightGBMParams{
		MaxDepth:        6,
		MaxLeaves:       31,
		MinChildSamples: 5,
		Estimators:      200,
		LearningRate:    0.05,
		L2LeafReg:       0.0,
	}

	trees, baseVal := TrainLeafwiseGBDT(XTrain, YTrain, WTrain, params)
	if len(trees) != params.Estimators {
		t.Fatalf("expected %d trees, got %d", params.Estimators, len(trees))
	}

	// 1. Every tree in the ensemble is valid: no node has SplitFeature != -1 with LeftChild == -1 || RightChild == -1.
	// 2. All leaf values remain strictly bounded (|LeafValue| < 1000.0).
	for tIdx, tree := range trees {
		for _, node := range tree.Nodes {
			if node.SplitFeature != -1 {
				if node.LeftChild == -1 || node.RightChild == -1 {
					t.Fatalf("tree %d node %d has SplitFeature=%d but corrupted children (leftChild=%d, rightChild=%d)",
						tIdx, node.NodeID, node.SplitFeature, node.LeftChild, node.RightChild)
				}
			} else {
				if math.IsNaN(node.LeafValue) || math.IsInf(node.LeafValue, 0) {
					t.Fatalf("tree %d node %d has non-finite leaf value: %f", tIdx, node.NodeID, node.LeafValue)
				}
				if math.Abs(node.LeafValue) >= 1000.0 {
					t.Errorf("tree %d node %d leaf value exploded: %f", tIdx, node.NodeID, node.LeafValue)
				}
			}
		}
	}

	// 3. Predictions on training and holdout samples do not contain NaN, Inf, or exploding values.
	checkPredictions := func(name string, samples []FeatureVector) {
		for i, fv := range samples {
			pred := baseVal
			for _, tree := range trees {
				pred += params.LearningRate * EvaluateTree(tree.Nodes, fv)
			}
			if math.IsNaN(pred) || math.IsInf(pred, 0) {
				t.Fatalf("%s prediction %d is non-finite: %f", name, i, pred)
			}
			if math.Abs(pred) > 1000.0 {
				t.Errorf("%s prediction %d exploded: %f", name, i, pred)
			}
		}
	}

	checkPredictions("training", XTrain)
	checkPredictions("holdout", XHoldout)
}

func TestExtraTrees_DenominatorRegularization_NearDegenerateWeights(t *testing.T) {
	// Verify that ExtraTrees with weights on the order of 1e-9 do not explode to 10^9
	nSamples := 50
	X := make([]FeatureVector, nSamples)
	Y := make([]float64, nSamples)
	W := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		X[i] = FeatureVector{Values: []float64{float64(i)}}
		Y[i] = 100.0 + float64(i%5)*10.0
		// Weights near 1e-9
		W[i] = 1e-9 + float64(i)*1e-10
	}

	params := ExtraTreesParams{
		Estimators:     20,
		MinSamplesLeaf: 2,
		MaxFeatures:    1.0,
		MaxLeafValue:   DefaultMaxLeafValue,
	}

	trees := TrainExtraTrees(X, Y, W, params)
	if len(trees) != 20 {
		t.Fatalf("expected 20 trees, got %d", len(trees))
	}

	for tIdx, tree := range trees {
		for _, node := range tree.Nodes {
			if node.SplitFeature == -1 {
				if math.IsNaN(node.LeafValue) || math.IsInf(node.LeafValue, 0) {
					t.Fatalf("tree %d node %d has non-finite leaf value: %f", tIdx, node.NodeID, node.LeafValue)
				}
				if math.Abs(node.LeafValue) > DefaultMaxLeafValue {
					t.Errorf("tree %d node %d leaf value exploded beyond MaxLeafValue: %f", tIdx, node.NodeID, node.LeafValue)
				}
			}
		}
	}
}

func TestLeafwiseGBDT_DegenerateSplit_BoundedLeafValues(t *testing.T) {
	// Construct degenerate dataset where most samples have zero/near-zero weight
	nSamples := 60
	X := make([]FeatureVector, nSamples)
	Y := make([]float64, nSamples)
	W := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		X[i] = FeatureVector{Values: []float64{float64(i % 3)}}
		if i < 3 {
			Y[i] = 50.0
			W[i] = 1.0
		} else {
			Y[i] = 0.0
			W[i] = 1e-9 // Near-degenerate child weights
		}
	}

	lgbParams := LightGBMParams{
		MaxDepth:        4,
		MaxLeaves:       8,
		MinChildSamples: 1,
		Estimators:      25,
		LearningRate:    0.1,
		L2LeafReg:       0.0,
		MaxLeafValue:    1e4,
	}

	lgbTrees, lgbBase := TrainLeafwiseGBDT(X, Y, W, lgbParams)
	if math.IsNaN(lgbBase) || math.IsInf(lgbBase, 0) {
		t.Fatalf("lgbBase is non-finite: %f", lgbBase)
	}

	for tIdx, tree := range lgbTrees {
		for _, node := range tree.Nodes {
			if node.SplitFeature == -1 {
				if math.IsNaN(node.LeafValue) || math.IsInf(node.LeafValue, 0) {
					t.Fatalf("tree %d node %d has non-finite leaf value: %f", tIdx, node.NodeID, node.LeafValue)
				}
				if math.Abs(node.LeafValue) > 1e4 {
					t.Errorf("tree %d node %d leaf value exceeded bound: %f", tIdx, node.NodeID, node.LeafValue)
				}
			}
		}
	}

	cbParams := CatBoostParams{
		Depth:        3,
		Iterations:   25,
		LearningRate: 0.1,
		L2LeafReg:    0.0,
		MaxLeafValue: 1e4,
	}
	cbTrees, cbBase := TrainSymmetricGBDT(X, Y, W, cbParams)
	if math.IsNaN(cbBase) || math.IsInf(cbBase, 0) {
		t.Fatalf("cbBase is non-finite: %f", cbBase)
	}

	for tIdx, tree := range cbTrees {
		for _, node := range tree.Nodes {
			if node.SplitFeature == -1 {
				if math.IsNaN(node.LeafValue) || math.IsInf(node.LeafValue, 0) {
					t.Fatalf("catboost tree %d node %d has non-finite leaf value: %f", tIdx, node.NodeID, node.LeafValue)
				}
				if math.Abs(node.LeafValue) > 1e4 {
					t.Errorf("catboost tree %d node %d leaf value exceeded bound: %f", tIdx, node.NodeID, node.LeafValue)
				}
			}
		}
	}
}

func TestEnsemble_ConfigurableMaxLeafValue(t *testing.T) {
	nSamples := 40
	X := make([]FeatureVector, nSamples)
	Y := make([]float64, nSamples)
	W := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		X[i] = FeatureVector{Values: []float64{float64(i)}}
		Y[i] = float64(i * 10)
		W[i] = 1.0
	}

	cfg := DefaultEnsembleConfig()
	cfg.MaxLeafValue = 15.0
	cfg.CatBoost.Iterations = 10
	cfg.LightGBM.Estimators = 10
	cfg.ExtraTrees.Estimators = 10

	model := TrainEnsembleWithWeights(X, Y, W, cfg)
	if model == nil {
		t.Fatalf("TrainEnsembleWithWeights returned nil")
	}

	for tIdx, tree := range model.Trees {
		for _, node := range tree.Nodes {
			if node.SplitFeature == -1 {
				if math.Abs(node.LeafValue) > 15.0 {
					t.Errorf("tree %d node %d leaf value %f exceeded configured MaxLeafValue 15.0",
						tIdx, node.NodeID, node.LeafValue)
				}
			}
		}
	}
}

func TestExtraTrees_DenominatorRegularization_Continuity(t *testing.T) {
	// Verify that weights around 1.0 (e.g. 0.999 vs 1.001) behave continuously without step jumps
	nSamples := 40
	X := make([]FeatureVector, nSamples)
	Y := make([]float64, nSamples)
	wA := make([]float64, nSamples)
	wB := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		X[i] = FeatureVector{Values: []float64{float64(i)}}
		Y[i] = 100.0
		wA[i] = 0.999
		wB[i] = 1.001
	}

	params := ExtraTreesParams{
		Estimators:     10,
		MinSamplesLeaf: 1,
		MaxFeatures:    1.0,
		MaxLeafValue:   DefaultMaxLeafValue,
	}

	treesA := TrainExtraTrees(X, Y, wA, params)
	treesB := TrainExtraTrees(X, Y, wB, params)

	leafA := treesA[0].Nodes[len(treesA[0].Nodes)-1].LeafValue
	leafB := treesB[0].Nodes[len(treesB[0].Nodes)-1].LeafValue

	// Relative difference between 0.999 and 1.001 should be well under 5%, never a 50% step jump
	relDiff := math.Abs(leafA-leafB) / math.Max(1.0, math.Abs(leafA))
	if relDiff > 0.05 {
		t.Errorf("discontinuous leaf values across weight boundary 1.0: leafA=%f, leafB=%f, relDiff=%f",
			leafA, leafB, relDiff)
	}
}

func TestEnsemble_SubUnitFractionalWeights(t *testing.T) {
	nSamples := 50
	X := make([]FeatureVector, nSamples)
	Y := make([]float64, nSamples)
	W := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		X[i] = FeatureVector{Values: []float64{float64(i)}}
		Y[i] = 50.0 + float64(i)
		W[i] = 0.005 // total sum of weights is 0.25 (<< 1.0)
	}

	cfg := DefaultEnsembleConfig()
	cfg.CatBoost.Iterations = 10
	cfg.LightGBM.Estimators = 10
	cfg.ExtraTrees.Estimators = 10

	model := TrainEnsembleWithWeights(X, Y, W, cfg)
	if model == nil {
		t.Fatalf("TrainEnsembleWithWeights returned nil")
	}

	for _, tree := range model.Trees {
		for _, node := range tree.Nodes {
			if node.SplitFeature == -1 {
				if math.IsNaN(node.LeafValue) || math.IsInf(node.LeafValue, 0) {
					t.Fatalf("non-finite leaf value under fractional weights: %f", node.LeafValue)
				}
				if math.Abs(node.LeafValue) > DefaultMaxLeafValue {
					t.Fatalf("leaf value exploded under fractional weights: %f", node.LeafValue)
				}
			}
		}
	}

	pred := model.Predict([]float64{25.0})
	if math.IsNaN(pred) || math.IsInf(pred, 0) {
		t.Fatalf("non-finite prediction under fractional weights: %f", pred)
	}
}

func TestEnsemble_NegativeAndDegenerateHyperparameters(t *testing.T) {
	nSamples := 20
	X := make([]FeatureVector, nSamples)
	Y := make([]float64, nSamples)
	W := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		X[i] = FeatureVector{Values: []float64{float64(i)}}
		Y[i] = float64(i * 5)
		W[i] = 1.0
	}

	// 1. CatBoost with negative iterations, negative depth, and negative/NaN L2LeafReg
	cbParamsNeg := CatBoostParams{
		Depth:        -2,
		Iterations:   -5,
		LearningRate: 0.1,
		L2LeafReg:    -10.0,
	}
	treesCB, baseCB := TrainSymmetricGBDT(X, Y, W, cbParamsNeg)
	if len(treesCB) != 0 {
		t.Errorf("expected 0 trees for negative iterations, got %d", len(treesCB))
	}
	if math.IsNaN(baseCB) || math.IsInf(baseCB, 0) {
		t.Errorf("expected finite baseValue, got %f", baseCB)
	}

	cbParamsValidIter := CatBoostParams{
		Depth:        -1,
		Iterations:   5,
		LearningRate: 0.1,
		L2LeafReg:    math.NaN(),
	}
	treesCB2, _ := TrainSymmetricGBDT(X, Y, W, cbParamsValidIter)
	if len(treesCB2) != 5 {
		t.Errorf("expected 5 trees for clamped depth, got %d", len(treesCB2))
	}

	// 2. LightGBM with negative estimators, negative maxDepth, negative minChildSamples, negative/NaN L2LeafReg
	lgbParamsNeg := LightGBMParams{
		MaxDepth:        -2,
		MaxLeaves:       -4,
		MinChildSamples: -3,
		Estimators:      -5,
		LearningRate:    0.1,
		L2LeafReg:       -10.0,
	}
	treesLGB, baseLGB := TrainLeafwiseGBDT(X, Y, W, lgbParamsNeg)
	if len(treesLGB) != 0 {
		t.Errorf("expected 0 trees for negative estimators, got %d", len(treesLGB))
	}
	if math.IsNaN(baseLGB) || math.IsInf(baseLGB, 0) {
		t.Errorf("expected finite baseValue, got %f", baseLGB)
	}

	lgbParamsValidEst := LightGBMParams{
		MaxDepth:        -1,
		MaxLeaves:       -1,
		MinChildSamples: -1,
		Estimators:      5,
		LearningRate:    0.1,
		L2LeafReg:       math.NaN(),
	}
	treesLGB2, _ := TrainLeafwiseGBDT(X, Y, W, lgbParamsValidEst)
	if len(treesLGB2) != 5 {
		t.Errorf("expected 5 trees for clamped params, got %d", len(treesLGB2))
	}

	// 3. ExtraTrees with negative estimators, negative minChildSamples, negative/NaN maxFeatures
	etParamsNeg := ExtraTreesParams{
		Estimators:     -5,
		MinSamplesLeaf: -3,
		MaxFeatures:    -0.5,
	}
	treesET := TrainExtraTrees(X, Y, W, etParamsNeg)
	if len(treesET) != 0 {
		t.Errorf("expected 0 trees for negative estimators, got %d", len(treesET))
	}

	etParamsValidEst := ExtraTreesParams{
		Estimators:     5,
		MinSamplesLeaf: -1,
		MaxFeatures:    math.NaN(),
	}
	treesET2 := TrainExtraTrees(X, Y, W, etParamsValidEst)
	if len(treesET2) != 5 {
		t.Errorf("expected 5 trees for clamped params, got %d", len(treesET2))
	}

	// 4. Combined ensemble with negative/degenerate settings
	cfg := EnsembleConfig{
		MaxLeafValue:     1e4,
		CatBoostWeight:   1.0,
		LightGBMWeight:   1.0,
		ExtraTreesWeight: 1.0,
		CatBoost:         cbParamsValidIter,
		LightGBM:         lgbParamsValidEst,
		ExtraTrees:       etParamsValidEst,
	}
	model := TrainEnsembleWithWeights(X, Y, W, cfg)
	if model == nil {
		t.Fatalf("expected non-nil model for ensemble with degenerate settings")
	}
	pred := model.Predict([]float64{10.0})
	if math.IsNaN(pred) || math.IsInf(pred, 0) {
		t.Errorf("expected finite prediction, got %f", pred)
	}
}

// FuzzTrainEnsembleWithWeights tests ensemble training against arbitrary targets and sample weights.
func FuzzTrainEnsembleWithWeights(f *testing.F) {
	f.Add(float64(10.0), float64(50.0), float64(1.0), float64(10.0), float64(500.0))
	f.Add(float64(-100.0), float64(100.0), float64(1e-9), float64(1e-6), float64(10.0))
	f.Add(float64(0.0), float64(0.0), float64(0.0), float64(0.0), float64(1e4))

	f.Fuzz(func(t *testing.T, y0, y1, w0, w1, maxLeaf float64) {
		if math.IsNaN(y0) || math.IsNaN(y1) || math.IsNaN(w0) || math.IsNaN(w1) || math.IsNaN(maxLeaf) {
			return
		}
		if math.IsInf(y0, 0) || math.IsInf(y1, 0) || math.IsInf(w0, 0) || math.IsInf(w1, 0) || math.IsInf(maxLeaf, 0) {
			return
		}

		X := []FeatureVector{
			{Values: []float64{1.0, 2.0}},
			{Values: []float64{3.0, 4.0}},
			{Values: []float64{5.0, 6.0}},
			{Values: []float64{7.0, 8.0}},
		}
		Y := []float64{y0, y1, (y0 + y1) / 2.0, y1 * 2.0}
		W := []float64{math.Abs(w0), math.Abs(w1), 1.0, 2.0}

		cfg := DefaultEnsembleConfig()
		cfg.MaxLeafValue = math.Abs(maxLeaf)
		cfg.CatBoost.Iterations = 2
		cfg.LightGBM.Estimators = 2
		cfg.ExtraTrees.Estimators = 2

		model := TrainEnsembleWithWeights(X, Y, W, cfg)
		if model != nil {
			resolvedBound := cfg.MaxLeafValue
			if resolvedBound <= 0.0 {
				resolvedBound = DefaultMaxLeafValue
			}
			for _, tree := range model.Trees {
				for _, node := range tree.Nodes {
					if node.SplitFeature == -1 {
						if math.IsNaN(node.LeafValue) || math.IsInf(node.LeafValue, 0) {
							t.Fatalf("non-finite leaf value in trained ensemble: %f", node.LeafValue)
						}
						if math.Abs(node.LeafValue) > resolvedBound {
							t.Fatalf("leaf value %f exceeded bound %f", node.LeafValue, resolvedBound)
						}
					}
				}
			}
			pred := model.Predict([]float64{2.0, 3.0})
			if math.IsNaN(pred) || math.IsInf(pred, 0) {
				t.Fatalf("non-finite prediction: %f", pred)
			}
		}
	})
}
