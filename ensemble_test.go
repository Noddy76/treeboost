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
