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
	"time"
)

func TestChronologicalCV(t *testing.T) {
	nSamples := 120
	baseTime := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	X := make([]FeatureVector, nSamples)
	Y := make([]float64, nSamples)
	timestamps := make([]time.Time, nSamples)

	for i := 0; i < nSamples; i++ {
		timestamps[i] = baseTime.Add(time.Duration(i) * time.Hour)
		val1 := float64(i) / 10.0
		val2 := math.Sin(float64(i) * 0.1)
		val3 := 0.0 // lead time feature slot
		X[i] = FeatureVector{Values: []float64{val1, val2, val3}}
		Y[i] = 10.0 + 2.0*val1 + 5.0*val2 + math.Sin(float64(i))
	}

	// Keep a copy of X[100].Values[2] to verify X is not mutated during CV
	originalVal := X[100].Values[2]

	config := DefaultEnsembleConfig()
	config.CatBoost.Iterations = 20
	config.LightGBM.Estimators = 20
	config.ExtraTrees.Estimators = 20

	numFolds := 4
	leadTimeFeatureIndex := 2

	offsets, metrics, err := ChronologicalCV(X, Y, timestamps, config, numFolds, leadTimeFeatureIndex)
	if err != nil {
		t.Fatalf("unexpected error from ChronologicalCV: %v", err)
	}

	// 1. Verify X was not mutated
	if X[100].Values[2] != originalVal {
		t.Fatalf("X feature was mutated during validation: expected %f, got %f", originalVal, X[100].Values[2])
	}

	// 2. Check metrics
	if metrics.ValidationMAE <= 0 {
		t.Errorf("expected ValidationMAE > 0, got %f", metrics.ValidationMAE)
	}
	if metrics.ValidationRMSE <= 0 {
		t.Errorf("expected ValidationRMSE > 0, got %f", metrics.ValidationRMSE)
	}
	if math.IsNaN(metrics.RSquared) {
		t.Errorf("expected non-NaN RSquared, got %f", metrics.RSquared)
	}

	// 3. Verify offsets length matches DefaultHorizonBins
	expectedBins := len(DefaultHorizonBins)
	if len(offsets.P10) != expectedBins || len(offsets.P90) != expectedBins {
		t.Fatalf("offsets length mismatch: P10=%d, P90=%d, expected=%d",
			len(offsets.P10), len(offsets.P90), expectedBins)
	}

	// 4. Test monotonic envelope expansion:
	// P10[b] <= P10[b-1] <= 0 and P90[b] >= P90[b-1] >= 0
	if offsets.P10[0] > 0 {
		t.Errorf("expected P10[0] <= 0, got %f", offsets.P10[0])
	}
	if offsets.P90[0] < 0 {
		t.Errorf("expected P90[0] >= 0, got %f", offsets.P90[0])
	}

	for b := 1; b < expectedBins; b++ {
		if offsets.P10[b] > offsets.P10[b-1] {
			t.Errorf("P10 non-monotonic expansion at bin %d: P10[%d]=%f > P10[%d]=%f",
				b, b, offsets.P10[b], b-1, offsets.P10[b-1])
		}
		if offsets.P10[b] > 0 {
			t.Errorf("P10 at bin %d must be <= 0, got %f", b, offsets.P10[b])
		}

		if offsets.P90[b] < offsets.P90[b-1] {
			t.Errorf("P90 non-monotonic expansion at bin %d: P90[%d]=%f < P90[%d]=%f",
				b, b, offsets.P90[b], b-1, offsets.P90[b-1])
		}
		if offsets.P90[b] < 0 {
			t.Errorf("P90 at bin %d must be >= 0, got %f", b, offsets.P90[b])
		}
	}
}

func TestChronologicalCV_EdgeCases(t *testing.T) {
	config := DefaultEnsembleConfig()

	// Empty dataset
	if _, _, err := ChronologicalCV(nil, nil, nil, config, 3, -1); err == nil {
		t.Error("expected error for empty dataset, got nil")
	}

	// Mismatched lengths
	X := make([]FeatureVector, 10)
	Y := make([]float64, 9)
	ts := make([]time.Time, 10)
	if _, _, err := ChronologicalCV(X, Y, ts, config, 3, -1); err == nil {
		t.Error("expected error for mismatched X and Y lengths, got nil")
	}

	Y = make([]float64, 10)
	ts = make([]time.Time, 8)
	if _, _, err := ChronologicalCV(X, Y, ts, config, 3, -1); err == nil {
		t.Error("expected error for mismatched timestamps length, got nil")
	}

	// numFolds < 2
	ts = make([]time.Time, 10)
	if _, _, err := ChronologicalCV(X, Y, ts, config, 1, -1); err == nil {
		t.Error("expected error for numFolds < 2, got nil")
	}

	// numFolds > len(X)
	if _, _, err := ChronologicalCV(X, Y, ts, config, 15, -1); err == nil {
		t.Error("expected error for numFolds > len(X), got nil")
	}

	// leadTimeFeatureIndex out of range
	for i := range X {
		X[i] = FeatureVector{Values: []float64{1.0, 2.0}}
	}
	if _, _, err := ChronologicalCV(X, Y, ts, config, 2, 5); err == nil {
		t.Error("expected error for out of bounds leadTimeFeatureIndex, got nil")
	}
}

func TestChronologicalCVWithOptions_MaxLeadHours(t *testing.T) {
	// Create a chronological dataset spanning 20 days with hourly intervals: 20 * 24 = 480 samples.
	const days = 20
	const hoursPerDay = 24
	nSamples := days * hoursPerDay
	baseTime := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	X := make([]FeatureVector, nSamples)
	Y := make([]float64, nSamples)
	timestamps := make([]time.Time, nSamples)

	for i := 0; i < nSamples; i++ {
		timestamps[i] = baseTime.Add(time.Duration(i) * time.Hour)
		val := float64(i) * 0.1
		X[i] = FeatureVector{Values: []float64{val, math.Sin(val)}}
		Y[i] = 15.0 + 3.0*val + 2.0*math.Sin(val)
	}

	config := DefaultEnsembleConfig()
	config.CatBoost.Iterations = 10
	config.LightGBM.Estimators = 10
	config.ExtraTrees.Estimators = 10

	// Run with 2 folds:
	// Fold 0: 0..239 (10 days) -> Train
	// Fold 1: 240..479 (10 days = 240 hours) -> Validation
	// Forecast origin: timestamps[239].
	// Lead times in validation range from 1.0 hour to 240.0 hours.

	// 1. Uncapped CV: MaxLeadHours = 0.0 (all 240 samples in fold 1 evaluated)
	uncappedOpts := ChronologicalCVOptions{
		NumFolds:             2,
		LeadTimeFeatureIndex: -1,
		MaxLeadHours:         0.0,
		HorizonBins:          ExtendedHorizonBins,
	}
	_, uncappedMetrics, err := ChronologicalCVWithOptions(X, Y, timestamps, config, uncappedOpts)
	if err != nil {
		t.Fatalf("unexpected error for uncapped CV: %v", err)
	}
	if uncappedMetrics.ValidationSamples != 240 {
		t.Errorf("expected 240 uncapped validation samples, got %d", uncappedMetrics.ValidationSamples)
	}

	// 2. Capped CV: MaxLeadHours = 120.0 (5 days = 120 hours)
	// Samples 240..359 have leadHours <= 120.0 (120 samples)
	// Samples 360..479 have leadHours > 120.0 (120 samples skipped)
	cappedOpts := ChronologicalCVOptions{
		NumFolds:             2,
		LeadTimeFeatureIndex: -1,
		MaxLeadHours:         120.0,
		HorizonBins:          ExtendedHorizonBins,
	}
	offsets, cappedMetrics, err := ChronologicalCVWithOptions(X, Y, timestamps, config, cappedOpts)
	if err != nil {
		t.Fatalf("unexpected error for capped CV: %v", err)
	}
	if cappedMetrics.ValidationSamples != 120 {
		t.Errorf("expected exactly 120 capped validation samples, got %d", cappedMetrics.ValidationSamples)
	}

	// 3. Verify monotonic expansion across all bins
	for b := 1; b < len(ExtendedHorizonBins); b++ {
		if offsets.P10[b] > offsets.P10[b-1] {
			t.Errorf("P10 non-monotonic at bin %d: P10[%d]=%f > P10[%d]=%f",
				b, b, offsets.P10[b], b-1, offsets.P10[b-1])
		}
		if offsets.P90[b] < offsets.P90[b-1] {
			t.Errorf("P90 non-monotonic at bin %d: P90[%d]=%f < P90[%d]=%f",
				b, b, offsets.P90[b], b-1, offsets.P90[b-1])
		}
	}
}

func TestChronologicalCVWithOptions_InferenceOverrides(t *testing.T) {
	const nSamples = 200
	baseTime := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	X := make([]FeatureVector, nSamples)
	Y := make([]float64, nSamples)
	timestamps := make([]time.Time, nSamples)

	for i := 0; i < nSamples; i++ {
		timestamps[i] = baseTime.Add(time.Duration(i) * time.Hour)
		val1 := float64(i) * 0.1
		val2 := math.Cos(float64(i) * 0.2)
		leadSlot := 0.0
		val4 := 1.0
		val5 := 2.0
		// Feature index 5: dummy training-only sample ages (10.0 ... 50.0)
		sampleAge := 10.0 + float64(i%40)
		X[i] = FeatureVector{Values: []float64{val1, val2, leadSlot, val4, val5, sampleAge}}
		// Target heavily dependent on feature index 5
		Y[i] = 10.0 + 2.0*val1 + 50.0*sampleAge
	}

	// Snapshot original feature index 5 to verify it is NOT mutated
	origSampleAges := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		origSampleAges[i] = X[i].Values[5]
	}

	config := DefaultEnsembleConfig()
	config.CatBoost.Iterations = 20
	config.LightGBM.Estimators = 20
	config.ExtraTrees.Estimators = 20

	// 1. Run CV with InferenceOverrides: index 5 overridden to 0.0 during validation
	overrideOpts := ChronologicalCVOptions{
		NumFolds:             3,
		LeadTimeFeatureIndex: 2,
		InferenceOverrides:   map[int]float64{5: 0.0},
	}
	_, overrideMetrics, err := ChronologicalCVWithOptions(X, Y, timestamps, config, overrideOpts)
	if err != nil {
		t.Fatalf("unexpected error with inference overrides: %v", err)
	}

	// Verify original X was not modified in-place
	for i := 0; i < nSamples; i++ {
		if X[i].Values[5] != origSampleAges[i] {
			t.Fatalf("X[%d].Values[5] was mutated: expected %f, got %f", i, origSampleAges[i], X[i].Values[5])
		}
	}

	// 2. Run CV without InferenceOverrides (feature index 5 is evaluated as 10.0 ... 50.0)
	noOverrideOpts := ChronologicalCVOptions{
		NumFolds:             3,
		LeadTimeFeatureIndex: 2,
	}
	_, noOverrideMetrics, err := ChronologicalCVWithOptions(X, Y, timestamps, config, noOverrideOpts)
	if err != nil {
		t.Fatalf("unexpected error without inference overrides: %v", err)
	}

	// Because Y heavily depends on feature 5 (coefficient 50.0 * sampleAge [10..50]),
	// overriding feature 5 to 0.0 produces a massive residual gap between the two evaluations.
	if math.Abs(overrideMetrics.ValidationMAE-noOverrideMetrics.ValidationMAE) < 100.0 {
		t.Errorf("expected significant difference in ValidationMAE when feature 5 is overridden to 0.0 vs unmasked, got override MAE=%f, unmasked MAE=%f",
			overrideMetrics.ValidationMAE, noOverrideMetrics.ValidationMAE)
	}
}

func TestChronologicalCVWithOptions_CustomHorizonBins(t *testing.T) {
	const nSamples = 120
	baseTime := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	X := make([]FeatureVector, nSamples)
	Y := make([]float64, nSamples)
	timestamps := make([]time.Time, nSamples)

	for i := 0; i < nSamples; i++ {
		timestamps[i] = baseTime.Add(time.Duration(i) * time.Hour)
		val := float64(i) * 0.1
		X[i] = FeatureVector{Values: []float64{val}}
		Y[i] = 10.0 + 2.0*val
	}

	config := DefaultEnsembleConfig()
	config.CatBoost.Iterations = 10
	config.LightGBM.Estimators = 10
	config.ExtraTrees.Estimators = 10

	// 1. Test 8-bin ExtendedHorizonBins
	extOpts := ChronologicalCVOptions{
		NumFolds:    3,
		HorizonBins: ExtendedHorizonBins,
	}
	extOffsets, _, err := ChronologicalCVWithOptions(X, Y, timestamps, config, extOpts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(extOffsets.P10) != len(ExtendedHorizonBins) || len(extOffsets.P90) != len(ExtendedHorizonBins) {
		t.Fatalf("expected %d bins for ExtendedHorizonBins, got P10=%d, P90=%d",
			len(ExtendedHorizonBins), len(extOffsets.P10), len(extOffsets.P90))
	}

	// 2. Test 4-bin custom slice
	customBins := []string{"0-12h", "12-24h", "24-48h", ">48h"}
	customOpts := ChronologicalCVOptions{
		NumFolds:    3,
		HorizonBins: customBins,
	}
	custOffsets, _, err := ChronologicalCVWithOptions(X, Y, timestamps, config, customOpts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(custOffsets.P10) != len(customBins) || len(custOffsets.P90) != len(customBins) {
		t.Fatalf("expected %d bins for customBins, got P10=%d, P90=%d",
			len(customBins), len(custOffsets.P10), len(custOffsets.P90))
	}
}

func TestChronologicalCVWithOptions_SampleWeightsConfig(t *testing.T) {
	const nSamples = 120
	baseTime := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	X := make([]FeatureVector, nSamples)
	Y := make([]float64, nSamples)
	timestamps := make([]time.Time, nSamples)

	for i := 0; i < nSamples; i++ {
		timestamps[i] = baseTime.Add(time.Duration(i) * time.Hour)
		val := float64(i) * 0.1
		X[i] = FeatureVector{Values: []float64{val}}
		if i%10 == 0 {
			Y[i] = -25.0
		} else {
			Y[i] = 20.0 + val
		}
	}

	config := DefaultEnsembleConfig()
	config.CatBoost.Iterations = 10
	config.LightGBM.Estimators = 10
	config.ExtraTrees.Estimators = 10

	customWeightCfg := WeightConfig{
		Multiplier:         8.0,
		Offset:             10.0,
		Subtrahend:         3.0,
		MinWeight:          1.0,
		NegativeMultiplier: 2.0,
	}

	opts := ChronologicalCVOptions{
		NumFolds:            3,
		SampleWeightsConfig: &customWeightCfg,
	}

	offsets, metrics, err := ChronologicalCVWithOptions(X, Y, timestamps, config, opts)
	if err != nil {
		t.Fatalf("unexpected error with SampleWeightsConfig: %v", err)
	}
	if metrics.ValidationMAE <= 0 {
		t.Errorf("expected positive ValidationMAE, got %f", metrics.ValidationMAE)
	}
	if len(offsets.P10) != len(DefaultHorizonBins) {
		t.Errorf("expected %d bins, got %d", len(DefaultHorizonBins), len(offsets.P10))
	}
}
