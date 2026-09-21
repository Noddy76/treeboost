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
)

// WeightConfig specifies parameters for sample weight computation.
type WeightConfig struct {
	Multiplier         float64 // default: 5.0
	Offset             float64 // default: 10.0
	Subtrahend         float64 // default: 4.0
	MinWeight          float64 // default: 1.0
	NegativeMultiplier float64 // default: 1.0 (e.g., set to 1.5 for 1.5x plunge penalty on y < 0)
}

// DefaultWeightConfig returns the default weighting parameters.
func DefaultWeightConfig() WeightConfig {
	return WeightConfig{
		Multiplier:         5.0,
		Offset:             10.0,
		Subtrahend:         4.0,
		MinWeight:          1.0,
		NegativeMultiplier: 1.0,
	}
}

// ComputeSampleWeightsWithConfig calculates weights for observations using a log-distance from the target mean
// according to the provided WeightConfig. If y < 0, the weight is scaled by cfg.NegativeMultiplier.
func ComputeSampleWeightsWithConfig(Y []float64, cfg WeightConfig) []float64 {
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
		w := cfg.Multiplier*math.Log10(diff+cfg.Offset) - cfg.Subtrahend
		w = math.Round(w)
		if w < cfg.MinWeight {
			w = cfg.MinWeight
		}
		if y < 0 {
			w *= cfg.NegativeMultiplier
		}
		weights[i] = w
	}
	return weights
}

// ComputeSampleWeights calculates weights for observations using default log-distance from target mean,
// emphasizing and penalizing extreme spikes/troughs.
func ComputeSampleWeights(Y []float64) []float64 {
	return ComputeSampleWeightsWithConfig(Y, DefaultWeightConfig())
}
