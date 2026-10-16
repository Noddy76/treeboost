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
	"path/filepath"
	"testing"
)

func TestWeightedQuantile_UniformAndEdgeCases(t *testing.T) {
	// Empty slice
	if q := WeightedQuantile(nil, nil, 0.5); q != 0.0 {
		t.Fatalf("expected 0.0 for nil slice, got %v", q)
	}
	if q := WeightedQuantile([]float64{}, []float64{}, 0.5); q != 0.0 {
		t.Fatalf("expected 0.0 for empty slice, got %v", q)
	}

	// Single sample
	if q := WeightedQuantile([]float64{42.0}, nil, 0.5); q != 42.0 {
		t.Fatalf("expected 42.0 for single sample, got %v", q)
	}
	if q := WeightedQuantile([]float64{42.0}, []float64{5.0}, 0.1); q != 42.0 {
		t.Fatalf("expected 42.0 for single sample with weight, got %v", q)
	}

	// All identical values
	if q := WeightedQuantile([]float64{7.0, 7.0, 7.0, 7.0}, nil, 0.3); q != 7.0 {
		t.Fatalf("expected 7.0 for identical values, got %v", q)
	}

	// Uniform sequence 1..5
	vals := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	q50 := WeightedQuantile(vals, nil, 0.5)
	if math.Abs(q50-3.0) > 1e-6 {
		t.Fatalf("expected median ~3.0 for uniform 1..5, got %v", q50)
	}

	q0 := WeightedQuantile(vals, nil, 0.0)
	if q0 != 1.0 {
		t.Fatalf("expected min 1.0 for q=0.0, got %v", q0)
	}

	q100 := WeightedQuantile(vals, nil, 1.0)
	if q100 != 5.0 {
		t.Fatalf("expected max 5.0 for q=1.0, got %v", q100)
	}

	// Quantile monotonicity: q10 <= q25 <= q50 <= q75 <= q90
	q10 := WeightedQuantile(vals, nil, 0.10)
	q25 := WeightedQuantile(vals, nil, 0.25)
	q75 := WeightedQuantile(vals, nil, 0.75)
	q90 := WeightedQuantile(vals, nil, 0.90)

	if !(q10 <= q25 && q25 <= q50 && q50 <= q75 && q75 <= q90) {
		t.Fatalf("expected monotonic quantiles, got: q10=%v, q25=%v, q50=%v, q75=%v, q90=%v", q10, q25, q50, q75, q90)
	}
}

func TestWeightedQuantile_SkewedWeights(t *testing.T) {
	vals := []float64{10.0, 100.0}

	// Heavy weight on 10.0
	wHeavyLeft := []float64{99.0, 1.0}
	qLeft := WeightedQuantile(vals, wHeavyLeft, 0.5)
	if qLeft > 20.0 {
		t.Fatalf("expected median pulled towards 10.0 with 99:1 weight, got %v", qLeft)
	}

	// Heavy weight on 100.0
	wHeavyRight := []float64{1.0, 99.0}
	qRight := WeightedQuantile(vals, wHeavyRight, 0.5)
	if qRight < 90.0 {
		t.Fatalf("expected median pulled towards 100.0 with 1:99 weight, got %v", qRight)
	}
}

func TestWeightedQuantile_NonFiniteAndSafety(t *testing.T) {
	// Non-finite values should be filtered
	vals := []float64{math.NaN(), 10.0, math.Inf(1), 20.0, math.Inf(-1), 30.0}
	q50 := WeightedQuantile(vals, nil, 0.5)
	if math.Abs(q50-20.0) > 1e-6 {
		t.Fatalf("expected median 20.0 filtering non-finite values, got %v", q50)
	}

	// Non-positive or non-finite weights should be filtered
	vals2 := []float64{10.0, 20.0, 30.0}
	weights := []float64{-5.0, 0.0, 10.0}
	qVal := WeightedQuantile(vals2, weights, 0.5)
	if qVal != 30.0 {
		t.Fatalf("expected only 30.0 to have valid weight, got %v", qVal)
	}

	// Out-of-range q should clamp or return safe value
	if q := WeightedQuantile([]float64{1.0, 2.0}, nil, -0.5); q != 1.0 {
		t.Fatalf("expected clamp to 1.0 for q < 0, got %v", q)
	}
	if q := WeightedQuantile([]float64{1.0, 2.0}, nil, 1.5); q != 2.0 {
		t.Fatalf("expected clamp to 2.0 for q > 1, got %v", q)
	}
	if q := WeightedQuantile([]float64{1.0, 2.0}, nil, math.NaN()); q != 0.0 {
		t.Fatalf("expected 0.0 for q=NaN, got %v", q)
	}
}

func TestPinballLoss_Analytic(t *testing.T) {
	// Empty inputs
	if loss := PinballLoss(nil, nil, nil, 0.5); loss != 0.0 {
		t.Fatalf("expected 0.0 on nil, got %v", loss)
	}
	if loss := PinballLoss([]float64{1.0}, []float64{1.0, 2.0}, nil, 0.5); loss != 0.0 {
		t.Fatalf("expected 0.0 on length mismatch, got %v", loss)
	}

	// Exact predictions -> loss = 0.0
	yTrue := []float64{10.0, 20.0, 30.0}
	yPred := []float64{10.0, 20.0, 30.0}
	if loss := PinballLoss(yTrue, yPred, nil, 0.5); loss != 0.0 {
		t.Fatalf("expected 0.0 for exact predictions, got %v", loss)
	}

	// At tau = 0.5, PinballLoss == 0.5 * MAE
	yPredOff := []float64{12.0, 18.0, 35.0} // errors: -2, +2, -5 => MAE = (2 + 2 + 5)/3 = 3.0
	loss50 := PinballLoss(yTrue, yPredOff, nil, 0.5)
	expectedLoss50 := 0.5 * 3.0
	if math.Abs(loss50-expectedLoss50) > 1e-9 {
		t.Fatalf("expected loss 1.5, got %v", loss50)
	}

	// Asymmetric penalties at tau = 0.9:
	// yTrue = 10, yPred = 5 (underprediction, diff = +5): loss = 0.9 * 5 = 4.5
	loss90Under := PinballLoss([]float64{10.0}, []float64{5.0}, nil, 0.9)
	if math.Abs(loss90Under-4.5) > 1e-9 {
		t.Fatalf("expected 4.5, got %v", loss90Under)
	}

	// yTrue = 10, yPred = 15 (overprediction, diff = -5): loss = (1 - 0.9) * 5 = 0.5
	loss90Over := PinballLoss([]float64{10.0}, []float64{15.0}, nil, 0.9)
	if math.Abs(loss90Over-0.5) > 1e-9 {
		t.Fatalf("expected 0.5, got %v", loss90Over)
	}
}

func TestFailFastValidation(t *testing.T) {
	invalidQuantiles := []float64{-0.5, 0.0, 1.0, 1.5, math.NaN(), math.Inf(1), math.Inf(-1)}

	for _, badQ := range invalidQuantiles {
		// validateQuantileParams directly
		assertPanics(t, fmt.Sprintf("validateQuantileParams(LossPinball, %v)", badQ), func() {
			validateQuantileParams(LossPinball, badQ)
		})

		// DefaultQuantileEnsembleConfig
		assertPanics(t, fmt.Sprintf("DefaultQuantileEnsembleConfig(%v)", badQ), func() {
			DefaultQuantileEnsembleConfig(badQ)
		})

		// TrainQuantileEnsemble
		assertPanics(t, fmt.Sprintf("TrainQuantileEnsemble(%v)", badQ), func() {
			TrainQuantileEnsemble([]FeatureVector{{Values: []float64{1.0}}}, []float64{1.0}, badQ, DefaultEnsembleConfig())
		})

		// TrainQuantileEnsembleWithWeights
		assertPanics(t, fmt.Sprintf("TrainQuantileEnsembleWithWeights(%v)", badQ), func() {
			TrainQuantileEnsembleWithWeights(
				[]FeatureVector{{Values: []float64{1.0}}},
				[]float64{1.0},
				[]float64{1.0},
				badQ,
				DefaultEnsembleConfig(),
			)
		})
	}

	// Length mismatches
	assertPanics(t, "TrainQuantileEnsemble length mismatch X and Y", func() {
		TrainQuantileEnsemble(
			[]FeatureVector{{Values: []float64{1.0}}, {Values: []float64{2.0}}},
			[]float64{1.0},
			0.5,
			DefaultEnsembleConfig(),
		)
	})

	assertPanics(t, "TrainQuantileEnsembleWithWeights length mismatch W and Y", func() {
		TrainQuantileEnsembleWithWeights(
			[]FeatureVector{{Values: []float64{1.0}}},
			[]float64{1.0},
			[]float64{1.0, 2.0},
			0.5,
			DefaultEnsembleConfig(),
		)
	})

	// Fail-fast validation with empty samples
	assertPanics(t, "TrainSymmetricGBDTColumnar empty samples invalid quantile", func() {
		TrainSymmetricGBDTColumnar(ColumnarDataset{NumSamples: 0}, CatBoostParams{
			LossFunction: LossPinball,
			Quantile:     -0.5,
		})
	})

	assertPanics(t, "TrainLeafwiseGBDTColumnar empty samples invalid quantile", func() {
		TrainLeafwiseGBDTColumnar(ColumnarDataset{NumSamples: 0}, LightGBMParams{
			LossFunction: LossPinball,
			Quantile:     1.5,
		})
	})
}

func TestWeightedQuantile_InfinitesimalAndExtremeWeights(t *testing.T) {
	vals := []float64{10.0, 20.0}

	// 1e-20 weights (equal weights => median must be 15.0)
	wSmall := []float64{1e-20, 1e-20}
	qSmall := WeightedQuantile(vals, wSmall, 0.5)
	if math.Abs(qSmall-15.0) > 1e-6 {
		t.Fatalf("expected median 15.0 for 1e-20 weights, got %v", qSmall)
	}

	// 1e-300 weights (equal weights => median must be 15.0)
	wTiny := []float64{1e-300, 1e-300}
	qTiny := WeightedQuantile(vals, wTiny, 0.5)
	if math.Abs(qTiny-15.0) > 1e-6 {
		t.Fatalf("expected median 15.0 for 1e-300 weights, got %v", qTiny)
	}

	// 1e308 weights (sum overflows float64 to +Inf without scaling => median must be 15.0)
	wHuge := []float64{1e308, 1e308}
	qHuge := WeightedQuantile(vals, wHuge, 0.5)
	if math.Abs(qHuge-15.0) > 1e-6 {
		t.Fatalf("expected median 15.0 for 1e308 weights, got %v", qHuge)
	}
}

func TestPinballLoss_ExtremeFloatBoundaries(t *testing.T) {
	// Extreme differences that overflow float64
	loss := PinballLoss([]float64{1e308}, []float64{-1e308}, nil, 0.5)
	if math.IsNaN(loss) || math.IsInf(loss, 0) {
		t.Fatalf("expected finite PinballLoss on extreme floats, got %v", loss)
	}
}

func TestModelValidate_NegativePinballLoss(t *testing.T) {
	m := &Model{
		BaseValue:    0.0,
		LearningRate: 1.0,
		LossFunction: LossPinball,
		Quantile:     0.5,
		PinballLoss:  -1.0,
	}
	if err := m.Validate(); err == nil {
		t.Fatal("expected Validate to reject negative PinballLoss, but got nil")
	}
}

func TestTrainQuantileEnsemble_WeightNormalizationWithExtraTrees(t *testing.T) {
	X := []FeatureVector{
		{Values: []float64{1.0, 2.0}},
		{Values: []float64{3.0, 4.0}},
		{Values: []float64{5.0, 6.0}},
	}
	Y := []float64{10.0, 20.0, 30.0}

	cfg := DefaultEnsembleConfig()
	cfg.LossFunction = LossPinball
	cfg.Quantile = 0.5
	cfg.CatBoostWeight = 1.0
	cfg.LightGBMWeight = 1.0
	cfg.ExtraTreesWeight = 1.0
	cfg.CatBoost.Iterations = 5
	cfg.LightGBM.Estimators = 5
	cfg.ExtraTrees.Estimators = 10
	cfg.ExtraTrees.MinSamplesLeaf = 1

	model := TrainQuantileEnsemble(X, Y, 0.5, cfg)
	if model == nil {
		t.Fatal("expected non-nil model")
	}
	if err := model.Validate(); err != nil {
		t.Fatalf("model validation failed: %v", err)
	}

	pred := model.Predict(X[1].Values)
	// X[1] has target 20.0 (the exact median). All 3 algorithms predict ~20.0.
	// An ensemble of median predictors with equal weights must predict ~20.0.
	if math.Abs(pred-20.0) > 3.0 {
		t.Fatalf("expected ensemble prediction ~20.0 for median sample, got %v (baseValue=%v)", pred, model.BaseValue)
	}
}

func assertPanics(t *testing.T, name string, f func()) {
	t.Helper()
	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("expected panic for %s, but executed normally", name)
		}
	}()
	f()
}

func TestPinballGBDT_MonotonicityAndCoverage(t *testing.T) {
	// Synthetic heteroskedastic, positively skewed log-return data:
	// x in [0.5, 4.0]
	// y = sin(x) + (0.2 + 0.5 * x) * (Exp(1) - 1)
	rng := rand.New(rand.NewSource(42))

	nTrain := 1500
	trainX := make([]FeatureVector, nTrain)
	trainY := make([]float64, nTrain)
	for i := 0; i < nTrain; i++ {
		x := 0.5 + rng.Float64()*(4.0-0.5)
		noise := rng.ExpFloat64() - 1.0 // mean 0, variance 1, positive skew
		y := math.Sin(x) + (0.2+0.5*x)*noise
		trainX[i] = FeatureVector{Values: []float64{x}}
		trainY[i] = y
	}

	nTest := 1000
	testX := make([]FeatureVector, nTest)
	testY := make([]float64, nTest)
	for i := 0; i < nTest; i++ {
		x := 0.5 + rng.Float64()*(4.0-0.5)
		noise := rng.ExpFloat64() - 1.0
		y := math.Sin(x) + (0.2+0.5*x)*noise
		testX[i] = FeatureVector{Values: []float64{x}}
		testY[i] = y
	}

	cfg10 := DefaultQuantileEnsembleConfig(0.10)
	cfg10.CatBoost.Iterations = 200
	cfg10.LightGBM.Estimators = 200
	model10 := TrainQuantileEnsemble(trainX, trainY, 0.10, cfg10)

	cfg50 := DefaultQuantileEnsembleConfig(0.50)
	cfg50.CatBoost.Iterations = 200
	cfg50.LightGBM.Estimators = 200
	model50 := TrainQuantileEnsemble(trainX, trainY, 0.50, cfg50)

	cfg90 := DefaultQuantileEnsembleConfig(0.90)
	cfg90.CatBoost.Iterations = 200
	cfg90.LightGBM.Estimators = 200
	model90 := TrainQuantileEnsemble(trainX, trainY, 0.90, cfg90)

	// Also train baseline MSE ensemble
	cfgMSE := DefaultEnsembleConfig()
	cfgMSE.CatBoost.Iterations = 200
	cfgMSE.LightGBM.Estimators = 200
	modelMSE := TrainEnsemble(trainX, trainY, cfgMSE)

	// Evaluate across test set
	monotonicViolations := 0
	inIntervalCount := 0
	belowQ10Count := 0
	aboveQ90Count := 0

	preds10 := make([]float64, nTest)
	preds50 := make([]float64, nTest)
	preds90 := make([]float64, nTest)
	predsMSE := make([]float64, nTest)

	for i := 0; i < nTest; i++ {
		p10 := model10.Predict(testX[i].Values)
		p50 := model50.Predict(testX[i].Values)
		p90 := model90.Predict(testX[i].Values)
		pMSE := modelMSE.Predict(testX[i].Values)

		preds10[i] = p10
		preds50[i] = p50
		preds90[i] = p90
		predsMSE[i] = pMSE

		// Monotonicity check
		if !(p10 <= p50 && p50 <= p90) {
			monotonicViolations++
		}

		y := testY[i]
		if y >= p10 && y <= p90 {
			inIntervalCount++
		}
		if y < p10 {
			belowQ10Count++
		}
		if y > p90 {
			aboveQ90Count++
		}
	}

	coverage := float64(inIntervalCount) / float64(nTest)
	belowRate := float64(belowQ10Count) / float64(nTest)
	aboveRate := float64(aboveQ90Count) / float64(nTest)

	t.Logf("Empirical Coverage [Q10, Q90]: %.2f%% (target 80%%)", coverage*100)
	t.Logf("Below Q10 Rate: %.2f%% (target 10%%)", belowRate*100)
	t.Logf("Above Q90 Rate: %.2f%% (target 10%%)", aboveRate*100)
	t.Logf("Monotonicity violations: %d / %d", monotonicViolations, nTest)

	// Acceptance criteria:
	// Coverage within +- 5% of target 80% (i.e. [75%, 85%])
	if coverage < 0.75 || coverage > 0.85 {
		t.Fatalf("empirical coverage %.2f%% outside expected [75%%, 85%%]", coverage*100)
	}

	// Tails reasonably close to 10%
	if belowRate < 0.05 || belowRate > 0.16 {
		t.Fatalf("below Q10 rate %.2f%% outside expected range", belowRate*100)
	}
	if aboveRate < 0.05 || aboveRate > 0.16 {
		t.Fatalf("above Q90 rate %.2f%% outside expected range", aboveRate*100)
	}

	// Quantile monotonicity: Q10 <= Q50 <= Q90
	if monotonicViolations > 0 {
		t.Fatalf("detected %d quantile monotonicity violations across %d evaluation points", monotonicViolations, nTest)
	}

	// Pinball loss comparison vs MSE model:
	// Quantile models should have strictly lower pinball loss for their respective quantiles
	pinball10_QModel := PinballLoss(testY, preds10, nil, 0.10)
	pinball10_MSEModel := PinballLoss(testY, predsMSE, nil, 0.10)
	if pinball10_QModel >= pinball10_MSEModel {
		t.Fatalf("expected Q10 model pinball loss (%v) < MSE model pinball loss (%v)", pinball10_QModel, pinball10_MSEModel)
	}

	pinball90_QModel := PinballLoss(testY, preds90, nil, 0.90)
	pinball90_MSEModel := PinballLoss(testY, predsMSE, nil, 0.90)
	if pinball90_QModel >= pinball90_MSEModel {
		t.Fatalf("expected Q90 model pinball loss (%v) < MSE model pinball loss (%v)", pinball90_QModel, pinball90_MSEModel)
	}
}

func TestExtraTrees_QuantileMode(t *testing.T) {
	rng := rand.New(rand.NewSource(123))
	n := 300
	X := make([]FeatureVector, n)
	Y := make([]float64, n)
	W := make([]float64, n)
	for i := 0; i < n; i++ {
		x := rng.Float64() * 10.0
		X[i] = FeatureVector{Values: []float64{x}}
		Y[i] = 2.0*x + rng.NormFloat64()
		W[i] = 1.0
	}

	params := ExtraTreesParams{
		Estimators:     50,
		MinSamplesLeaf: 5,
		MaxFeatures:    1.0,
		LossFunction:   LossPinball,
		Quantile:       0.5,
	}

	trees := TrainExtraTrees(X, Y, W, params)
	if len(trees) != 50 {
		t.Fatalf("expected 50 trees, got %d", len(trees))
	}

	// Verify trees predict finite values
	for _, tree := range trees {
		pred := tree.Evaluate([]float64{5.0})
		if math.IsNaN(pred) || math.IsInf(pred, 0) {
			t.Fatalf("non-finite ExtraTrees prediction: %v", pred)
		}
	}

	// Verify ensemble with ExtraTrees in pinball mode
	cfg := DefaultEnsembleConfig()
	cfg.LossFunction = LossPinball
	cfg.Quantile = 0.50
	cfg.CatBoost.Iterations = 20
	cfg.LightGBM.Estimators = 20
	cfg.ExtraTrees.Estimators = 20
	cfg.ExtraTrees.MinSamplesLeaf = 5

	model := TrainQuantileEnsemble(X, Y, 0.50, cfg)
	if model == nil {
		t.Fatal("expected non-nil model")
	}
	if model.LossFunction != LossPinball || model.Quantile != 0.50 {
		t.Fatalf("unexpected model loss/quantile metadata: %v, %v", model.LossFunction, model.Quantile)
	}
	pred := model.Predict([]float64{5.0})
	if math.IsNaN(pred) || math.IsInf(pred, 0) {
		t.Fatalf("non-finite ensemble prediction: %v", pred)
	}
}

func TestModelSerializationRoundtrip_Quantile(t *testing.T) {
	X := []FeatureVector{
		{Values: []float64{1.0, 2.0}},
		{Values: []float64{3.0, 4.0}},
		{Values: []float64{5.0, 6.0}},
	}
	Y := []float64{10.0, 20.0, 30.0}

	cfg := DefaultQuantileEnsembleConfig(0.90)
	cfg.CatBoost.Iterations = 10
	cfg.LightGBM.Estimators = 10

	model := TrainQuantileEnsemble(X, Y, 0.90, cfg)
	if model.LossFunction != LossPinball || model.Quantile != 0.90 {
		t.Fatalf("expected pinball metadata, got %s, %v", model.LossFunction, model.Quantile)
	}
	if model.PinballLoss <= 0.0 || math.IsNaN(model.PinballLoss) {
		t.Fatalf("expected positive PinballLoss, got %v", model.PinballLoss)
	}

	tmpDir := t.TempDir()
	modelPath := filepath.Join(tmpDir, "quantile_model.json")

	if err := model.SaveModel(modelPath); err != nil {
		t.Fatalf("SaveModel failed: %v", err)
	}

	loaded, err := LoadModel(modelPath)
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}

	if loaded.LossFunction != LossPinball {
		t.Fatalf("expected LossFunction %s, got %s", LossPinball, loaded.LossFunction)
	}
	if loaded.Quantile != 0.90 {
		t.Fatalf("expected Quantile 0.90, got %v", loaded.Quantile)
	}
	if math.Abs(loaded.PinballLoss-model.PinballLoss) > 1e-9 {
		t.Fatalf("expected PinballLoss %v, got %v", model.PinballLoss, loaded.PinballLoss)
	}

	// Verify predictions match exactly
	predOrig := model.Predict([]float64{2.0, 3.0})
	predLoaded := loaded.Predict([]float64{2.0, 3.0})
	if predOrig != predLoaded {
		t.Fatalf("prediction mismatch: orig=%v, loaded=%v", predOrig, predLoaded)
	}
}

// FuzzWeightedQuantile tests WeightedQuantile against adversarial and arbitrary inputs.
func FuzzWeightedQuantile(f *testing.F) {
	f.Add(1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 0.5)
	f.Add(-100.0, 0.0, 100.0, 0.1, 10.0, 1000.0, 0.1)
	f.Add(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.9)

	f.Fuzz(func(t *testing.T, v0, v1, v2, w0, w1, w2, q float64) {
		vals := []float64{v0, v1, v2}
		weights := []float64{w0, w1, w2}

		res := WeightedQuantile(vals, weights, q)
		if math.IsNaN(res) || math.IsInf(res, 0) {
			t.Fatalf("non-finite WeightedQuantile result: %v", res)
		}

		// Also test with nil weights
		resUniform := WeightedQuantile(vals, nil, q)
		if math.IsNaN(resUniform) || math.IsInf(resUniform, 0) {
			t.Fatalf("non-finite uniform WeightedQuantile result: %v", resUniform)
		}
	})
}

// FuzzPinballLoss tests PinballLoss against arbitrary float inputs.
func FuzzPinballLoss(f *testing.F) {
	f.Add(10.0, 12.0, 1.0, 0.5)
	f.Add(-50.0, 50.0, 2.0, 0.1)
	f.Add(0.0, 0.0, 1.0, 0.9)

	f.Fuzz(func(t *testing.T, yt, yp, w, q float64) {
		yTrue := []float64{yt}
		yPred := []float64{yp}
		weights := []float64{w}

		loss := PinballLoss(yTrue, yPred, weights, q)
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			t.Fatalf("non-finite PinballLoss: %v", loss)
		}
		if loss < 0.0 {
			t.Fatalf("negative PinballLoss: %v", loss)
		}
	})
}

// FuzzTrainQuantileEnsemble tests quantile ensemble training against arbitrary targets and weights.
func FuzzTrainQuantileEnsemble(f *testing.F) {
	f.Add(float64(10.0), float64(50.0), float64(1.0), float64(10.0), 0.5)
	f.Add(float64(-10.0), float64(10.0), float64(0.5), float64(2.0), 0.1)
	f.Add(float64(0.0), float64(1.0), float64(1.0), float64(1.0), 0.9)

	f.Fuzz(func(t *testing.T, y0, y1, w0, w1, q float64) {
		if math.IsNaN(y0) || math.IsNaN(y1) || math.IsNaN(w0) || math.IsNaN(w1) {
			return
		}
		if math.IsInf(y0, 0) || math.IsInf(y1, 0) || math.IsInf(w0, 0) || math.IsInf(w1, 0) {
			return
		}

		X := []FeatureVector{
			{Values: []float64{1.0, 2.0}},
			{Values: []float64{3.0, 4.0}},
			{Values: []float64{5.0, 6.0}},
			{Values: []float64{7.0, 8.0}},
		}
		Y := []float64{y0, y1, (y0 + y1) / 2.0, y1 * 1.5}
		W := []float64{math.Abs(w0) + 0.01, math.Abs(w1) + 0.01, 1.0, 1.0}

		if math.IsNaN(q) || math.IsInf(q, 0) || q <= 0.0 || q >= 1.0 {
			// Must panic
			defer func() {
				_ = recover()
			}()
			cfg := DefaultQuantileEnsembleConfig(0.5)
			_ = TrainQuantileEnsembleWithWeights(X, Y, W, q, cfg)
			t.Fatalf("expected panic for invalid quantile %v", q)
			return
		}

		cfg := DefaultQuantileEnsembleConfig(q)
		cfg.CatBoost.Iterations = 2
		cfg.LightGBM.Estimators = 2

		model := TrainQuantileEnsembleWithWeights(X, Y, W, q, cfg)
		if model != nil {
			if err := model.Validate(); err != nil {
				t.Fatalf("model validation failed: %v", err)
			}
			pred := model.Predict([]float64{2.0, 3.0})
			if math.IsNaN(pred) || math.IsInf(pred, 0) {
				t.Fatalf("non-finite prediction: %v", pred)
			}
		}
	})
}
