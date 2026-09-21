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
	"testing"
)

func TestHorizonBinForLeadTime(t *testing.T) {
	testCases := []struct {
		leadHours float64
		wantBin   int
	}{
		{-5.0, 0},
		{0.0, 0},
		{5.999, 0},
		{6.0, 1},
		{11.999, 1},
		{12.0, 2},
		{23.999, 2},
		{24.0, 3},
		{35.999, 3},
		{36.0, 4},
		{47.999, 4},
		{48.0, 5},
		{100.0, 5},
	}

	for _, tc := range testCases {
		got := HorizonBinForLeadTime(tc.leadHours)
		if got != tc.wantBin {
			t.Errorf("HorizonBinForLeadTime(%f) = %d, want %d", tc.leadHours, got, tc.wantBin)
		}
	}
}

func TestModel_PredictInterval(t *testing.T) {
	// Case 1: Positive prices with standard offsets
	modelPos := &Model{
		BaseValue:   50.0,
		P10Offsets:  []float64{-5.0, -8.0, -12.0, -15.0, -18.0, -22.0},
		P90Offsets:  []float64{6.0, 9.0, 14.0, 18.0, 21.0, 25.0},
		HorizonBins: DefaultHorizonBins,
	}

	p10, p50, p90 := modelPos.PredictInterval([]float64{1.0}, 2.0) // Bin 0
	if math.Abs(p50-50.0) > 1e-9 {
		t.Errorf("expected p50=50.0, got %f", p50)
	}
	if math.Abs(p10-45.0) > 1e-9 {
		t.Errorf("expected p10=45.0, got %f", p10)
	}
	if math.Abs(p90-56.0) > 1e-9 {
		t.Errorf("expected p90=56.0, got %f", p90)
	}
	if !(p10 <= p50 && p50 <= p90) {
		t.Errorf("interval ordering violated: p10=%f, p50=%f, p90=%f", p10, p50, p90)
	}

	// Case 2: Severe negative prices (e.g. -30.0)
	modelNeg := &Model{
		BaseValue:   -30.0,
		P10Offsets:  []float64{-10.0, -15.0, -20.0, -25.0, -30.0, -35.0},
		P90Offsets:  []float64{8.0, 12.0, 16.0, 20.0, 24.0, 28.0},
		HorizonBins: DefaultHorizonBins,
	}

	p10Neg, p50Neg, p90Neg := modelNeg.PredictInterval([]float64{1.0}, 15.0) // Bin 2
	if math.Abs(p50Neg - -30.0) > 1e-9 {
		t.Errorf("expected p50=-30.0, got %f", p50Neg)
	}
	if math.Abs(p10Neg - -50.0) > 1e-9 {
		t.Errorf("expected p10=-50.0, got %f", p10Neg)
	}
	if math.Abs(p90Neg - -14.0) > 1e-9 {
		t.Errorf("expected p90=-14.0, got %f", p90Neg)
	}
	if !(p10Neg <= p50Neg && p50Neg <= p90Neg) {
		t.Errorf("negative interval ordering violated: p10=%f, p50=%f, p90=%f", p10Neg, p50Neg, p90Neg)
	}

	// Case 3: Pathological inverted / flipped offsets (P10 > 0, P90 < 0, P90 < P10)
	modelPathological := &Model{
		BaseValue:   -30.0,
		P10Offsets:  []float64{15.0},  // Raw P10 would be -15.0 (> p50)
		P90Offsets:  []float64{-20.0}, // Raw P90 would be -50.0 (< p50)
		HorizonBins: DefaultHorizonBins,
	}

	p10Clamped, p50Clamped, p90Clamped := modelPathological.PredictInterval([]float64{1.0}, 2.0)
	if !(p10Clamped <= p50Clamped && p50Clamped <= p90Clamped) {
		t.Fatalf("unconditional ordering failed on pathological offsets: p10=%f, p50=%f, p90=%f",
			p10Clamped, p50Clamped, p90Clamped)
	}
	// Both must be clamped to p50
	if p10Clamped != -30.0 || p90Clamped != -30.0 {
		t.Errorf("expected clamped p10=-30.0, p90=-30.0, got p10=%f, p90=%f", p10Clamped, p90Clamped)
	}

	// Case 4: Missing/empty offsets
	modelEmpty := &Model{BaseValue: 12.0}
	p10Empty, p50Empty, p90Empty := modelEmpty.PredictInterval([]float64{1.0}, 10.0)
	if p10Empty != 12.0 || p50Empty != 12.0 || p90Empty != 12.0 {
		t.Errorf("expected missing offsets to yield point prediction 12.0, got p10=%f, p50=%f, p90=%f",
			p10Empty, p50Empty, p90Empty)
	}

	// Case 5: Nil model
	var nilModel *Model
	p10Nil, p50Nil, p90Nil := nilModel.PredictInterval([]float64{1.0}, 10.0)
	if p10Nil != 0.0 || p50Nil != 0.0 || p90Nil != 0.0 {
		t.Errorf("expected nil model to return 0.0, got %f, %f, %f", p10Nil, p50Nil, p90Nil)
	}
}

func TestModel_PredictBatchInterval(t *testing.T) {
	model := &Model{
		BaseValue:   20.0,
		P10Offsets:  []float64{-3.0, -6.0, -9.0, -12.0, -15.0, -18.0},
		P90Offsets:  []float64{4.0, 8.0, 12.0, 16.0, 20.0, 24.0},
		HorizonBins: DefaultHorizonBins,
	}

	flatFeatures := []float64{
		1.0, 2.0,
		3.0, 4.0,
		5.0, 6.0,
	}
	leadHours := []float64{2.0, 15.0, 50.0}

	p10s, p50s, p90s := model.PredictBatchInterval(flatFeatures, leadHours)
	if len(p10s) != 3 || len(p50s) != 3 || len(p90s) != 3 {
		t.Fatalf("expected 3 batch results, got %d, %d, %d", len(p10s), len(p50s), len(p90s))
	}

	for i := 0; i < 3; i++ {
		singleFeatures := flatFeatures[i*2 : (i+1)*2]
		p10, p50, p90 := model.PredictInterval(singleFeatures, leadHours[i])
		if math.Abs(p10s[i]-p10) > 1e-9 || math.Abs(p50s[i]-p50) > 1e-9 || math.Abs(p90s[i]-p90) > 1e-9 {
			t.Errorf("batch mismatch at index %d: batch=(%f, %f, %f) vs single=(%f, %f, %f)",
				i, p10s[i], p50s[i], p90s[i], p10, p50, p90)
		}
		if !(p10s[i] <= p50s[i] && p50s[i] <= p90s[i]) {
			t.Errorf("batch ordering violated at index %d: %f <= %f <= %f", i, p10s[i], p50s[i], p90s[i])
		}
	}

	// Empty batch
	p10Empty, p50Empty, p90Empty := model.PredictBatchInterval(nil, nil)
	if p10Empty != nil || p50Empty != nil || p90Empty != nil {
		t.Errorf("expected nil slices for empty batch")
	}
}

func TestTrainEnsembleWithWeights(t *testing.T) {
	nSamples := 50
	X := make([]FeatureVector, nSamples)
	Y := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		X[i] = FeatureVector{Values: []float64{float64(i)}}
		if i < 25 {
			Y[i] = 10.0
		} else {
			Y[i] = 100.0
		}
	}

	// Weights 1: heavily weigh the lower half
	w1 := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		if i < 25 {
			w1[i] = 100.0
		} else {
			w1[i] = 0.01
		}
	}

	// Weights 2: heavily weigh the upper half
	w2 := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		if i < 25 {
			w2[i] = 0.01
		} else {
			w2[i] = 100.0
		}
	}

	config := DefaultEnsembleConfig()
	config.CatBoost.Iterations = 30
	config.LightGBM.Estimators = 30
	config.ExtraTrees.Estimators = 30

	m1 := TrainEnsembleWithWeights(X, Y, w1, config)
	m2 := TrainEnsembleWithWeights(X, Y, w2, config)

	testSample := []float64{12.0}
	pred1 := m1.Predict(testSample)
	pred2 := m2.Predict(testSample)

	// Model 1 (weighted heavily towards 10.0) should predict a value closer to 10.0 than Model 2
	if math.Abs(pred1-pred2) < 1.0 {
		t.Errorf("expected distinct model predictions with different weights, got pred1=%f, pred2=%f",
			pred1, pred2)
	}

	// Verify TrainEnsemble backwards-compatible wrapper runs without error
	mDefault := TrainEnsemble(X, Y, config)
	if mDefault == nil || len(mDefault.Trees) == 0 {
		t.Fatal("TrainEnsemble returned invalid model")
	}
}
