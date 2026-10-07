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

func TestComputeSampleWeightsWithConfig(t *testing.T) {
	// Empty slice test
	if w := ComputeSampleWeightsWithConfig(nil, DefaultWeightConfig()); w != nil {
		t.Fatalf("expected nil for empty input, got %v", w)
	}

	targets := []float64{-50.0, -10.0, 0.0, 10.0, 50.0, 100.0}

	// 1. Verify DefaultWeightConfig matches ComputeSampleWeights
	defaultWeights := ComputeSampleWeights(targets)
	cfgWeights := ComputeSampleWeightsWithConfig(targets, DefaultWeightConfig())

	if len(defaultWeights) != len(cfgWeights) {
		t.Fatalf("length mismatch: %d vs %d", len(defaultWeights), len(cfgWeights))
	}
	for i := range defaultWeights {
		if math.Abs(defaultWeights[i]-cfgWeights[i]) > 1e-9 {
			t.Errorf("mismatch at index %d: default=%f, cfg=%f", i, defaultWeights[i], cfgWeights[i])
		}
	}

	// 2. Verify NegativeMultiplier = 1.5 scales negative targets while preserving non-negative targets
	cfgAsymm := DefaultWeightConfig()
	cfgAsymm.NegativeMultiplier = 1.5
	asymmWeights := ComputeSampleWeightsWithConfig(targets, cfgAsymm)

	for i, y := range targets {
		if y >= 0 {
			if math.Abs(asymmWeights[i]-defaultWeights[i]) > 1e-9 {
				t.Errorf("expected non-negative target %f at index %d to remain unchanged: got %f, want %f",
					y, i, asymmWeights[i], defaultWeights[i])
			}
		} else {
			expected := defaultWeights[i] * 1.5
			if math.Abs(asymmWeights[i]-expected) > 1e-9 {
				t.Errorf("expected negative target %f at index %d to be scaled 1.5x: got %f, want %f",
					y, i, asymmWeights[i], expected)
			}
		}
	}

	// 3. Test custom parameters
	cfgCustom := WeightConfig{
		Multiplier:         10.0,
		Offset:             5.0,
		Subtrahend:         2.0,
		MinWeight:          2.5,
		NegativeMultiplier: 2.0,
	}
	customWeights := ComputeSampleWeightsWithConfig(targets, cfgCustom)
	for i, w := range customWeights {
		if targets[i] >= 0 && w < cfgCustom.MinWeight {
			t.Errorf("expected weight >= minWeight %f, got %f", cfgCustom.MinWeight, w)
		}
		if targets[i] < 0 && w < cfgCustom.MinWeight*cfgCustom.NegativeMultiplier {
			t.Errorf("expected scaled weight >= %f, got %f", cfgCustom.MinWeight*cfgCustom.NegativeMultiplier, w)
		}
	}
}

// FuzzComputeSampleWeights fuzzes sample weight calculation under arbitrary targets and configurations.
func FuzzComputeSampleWeights(f *testing.F) {
	f.Add(float64(10.0), float64(5.0), float64(4.0), float64(1.0), float64(1.0), float64(50.0))
	f.Add(float64(-10.0), float64(-5.0), float64(0.0), float64(0.5), float64(2.0), float64(-100.0))
	f.Add(float64(0.0), float64(0.0), float64(0.0), float64(0.0), float64(0.0), float64(0.0))
	f.Add(float64(1e6), float64(1e6), float64(1e6), float64(100.0), float64(10.0), float64(1e12))

	f.Fuzz(func(t *testing.T, mult, offset, subtrahend, minW, negMult, targetVal float64) {
		cfg := WeightConfig{
			Multiplier:         mult,
			Offset:             offset,
			Subtrahend:         subtrahend,
			MinWeight:          minW,
			NegativeMultiplier: negMult,
		}

		targets := []float64{
			targetVal,
			-targetVal,
			0.0,
			targetVal * 2.5,
			math.NaN(),
			math.Inf(1),
			math.Inf(-1),
		}

		weights := ComputeSampleWeightsWithConfig(targets, cfg)
		if len(weights) != len(targets) {
			t.Fatalf("expected %d weights, got %d", len(targets), len(weights))
		}

		expectedMin := cfg.MinWeight
		if expectedMin <= 0.0 || math.IsNaN(expectedMin) || math.IsInf(expectedMin, 0) {
			expectedMin = 1.0
		}

		for i, w := range weights {
			if math.IsNaN(w) || math.IsInf(w, 0) {
				t.Fatalf("weight at index %d is non-finite: %f (target=%f)", i, w, targets[i])
			}
			if w < expectedMin {
				t.Fatalf("weight %f at index %d is less than minWeight %f (target=%f)", w, i, expectedMin, targets[i])
			}
		}
	})
}
