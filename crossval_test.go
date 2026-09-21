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
