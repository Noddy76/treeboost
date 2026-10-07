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
	"errors"
	"fmt"
	"math"
	"sort"
	"time"
)

// CVMetrics holds aggregated out-of-fold cross-validation performance metrics.
type CVMetrics struct {
	ValidationMAE     float64 `json:"validation_mae"`
	ValidationRMSE    float64 `json:"validation_rmse"`
	RSquared          float64 `json:"r_squared"`
	ValidationSamples int     `json:"validation_samples,omitempty"`
}

// HorizonOffsets holds empirical residual quantiles partitioned across horizon bins.
type HorizonOffsets struct {
	P10 []float64 `json:"p10"`
	P90 []float64 `json:"p90"`
}

// DefaultMaxValidPrediction defines the default ceiling on valid out-of-fold predictions (1e6).
const DefaultMaxValidPrediction = 1e6

// ChronologicalCVOptions configures chronological rolling-origin cross-validation.
type ChronologicalCVOptions struct {
	NumFolds             int
	LeadTimeFeatureIndex int             // Feature index where (slotTime - originTime)/24h is injected, or -1 to skip
	MaxLeadHours         float64         // Max forward horizon in hours to evaluate (<= 0.0 means unlimited)
	InferenceOverrides   map[int]float64 // Map of feature index -> constant value to override at validation time
	HorizonBins          []string        // Optional custom horizon bin names (defaults to DefaultHorizonBins if nil/empty)
	SampleWeightsConfig  *WeightConfig   // Optional sample weight config for fold training (nil uses DefaultWeightConfig)
	MaxValidPrediction   float64         // Max valid prediction value (defaults to 1e6 when unset or <= 0.0)
}

func linearQuantile(sorted []float64, q float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0.0
	}
	if n == 1 {
		return sorted[0]
	}
	idx := q * float64(n-1)
	lower := int(math.Floor(idx))
	upper := int(math.Ceil(idx))
	if lower == upper {
		return sorted[lower]
	}
	fraction := idx - float64(lower)
	return sorted[lower] + fraction*(sorted[upper]-sorted[lower])
}

// ChronologicalCVWithOptions performs forward-chaining rolling-origin cross-validation with custom options.
func ChronologicalCVWithOptions(
	X []FeatureVector,
	Y []float64,
	timestamps []time.Time,
	config EnsembleConfig,
	opts ChronologicalCVOptions,
) (HorizonOffsets, CVMetrics, error) {
	n := len(X)
	if n == 0 {
		return HorizonOffsets{}, CVMetrics{}, errors.New("empty dataset provided")
	}
	if len(Y) != n {
		return HorizonOffsets{}, CVMetrics{}, fmt.Errorf("length mismatch: len(X)=%d != len(Y)=%d", n, len(Y))
	}
	if len(timestamps) != n {
		return HorizonOffsets{}, CVMetrics{}, fmt.Errorf("length mismatch: len(X)=%d != len(timestamps)=%d", n, len(timestamps))
	}
	if opts.NumFolds < 2 {
		return HorizonOffsets{}, CVMetrics{}, fmt.Errorf("numFolds must be >= 2, got %d", opts.NumFolds)
	}
	if n < opts.NumFolds {
		return HorizonOffsets{}, CVMetrics{}, fmt.Errorf("number of samples (%d) is less than numFolds (%d)", n, opts.NumFolds)
	}
	if opts.LeadTimeFeatureIndex >= 0 && opts.LeadTimeFeatureIndex >= len(X[0].Values) {
		return HorizonOffsets{}, CVMetrics{}, fmt.Errorf("leadTimeFeatureIndex %d out of bounds (features len %d)", opts.LeadTimeFeatureIndex, len(X[0].Values))
	}

	maxValidPred := opts.MaxValidPrediction
	if maxValidPred <= 0.0 || math.IsNaN(maxValidPred) || math.IsInf(maxValidPred, 0) {
		maxValidPred = DefaultMaxValidPrediction
	}

	horizonBins := opts.HorizonBins
	if len(horizonBins) == 0 {
		horizonBins = DefaultHorizonBins
	}
	numBins := len(horizonBins)
	binResiduals := make([][]float64, numBins)

	var allValActuals []float64
	var allValPredictions []float64

	for k := 1; k < opts.NumFolds; k++ {
		trainEnd := (k * n) / opts.NumFolds
		valEnd := ((k + 1) * n) / opts.NumFolds
		if trainEnd <= 0 || trainEnd >= valEnd {
			continue
		}

		trainX := X[:trainEnd]
		trainY := Y[:trainEnd]

		var trainW []float64
		if opts.SampleWeightsConfig != nil {
			trainW = ComputeSampleWeightsWithConfig(trainY, *opts.SampleWeightsConfig)
		} else {
			trainW = ComputeSampleWeights(trainY)
		}
		model := TrainEnsembleWithWeights(trainX, trainY, trainW, config)

		originTime := timestamps[trainEnd-1]

		for i := trainEnd; i < valEnd; i++ {
			slotTime := timestamps[i]
			leadHours := math.Max(0.0, slotTime.Sub(originTime).Hours())

			if opts.MaxLeadHours > 0.0 && leadHours > opts.MaxLeadHours {
				continue
			}

			fvVals := make([]float64, len(X[i].Values))
			copy(fvVals, X[i].Values)
			if opts.LeadTimeFeatureIndex >= 0 && opts.LeadTimeFeatureIndex < len(fvVals) {
				fvVals[opts.LeadTimeFeatureIndex] = leadHours / 24.0
			}
			for fIdx, val := range opts.InferenceOverrides {
				if fIdx >= 0 && fIdx < len(fvVals) {
					fvVals[fIdx] = val
				}
			}

			pred := model.Predict(fvVals)
			if math.IsNaN(pred) || math.IsInf(pred, 0) || math.Abs(pred) > maxValidPred {
				return HorizonOffsets{}, CVMetrics{}, fmt.Errorf("chronological CV fold %d produced invalid out-of-fold prediction: %v", k, pred)
			}
			res := Y[i] - pred

			bin := HorizonBinForLeadTimeWithBins(leadHours, horizonBins)
			if bin >= numBins {
				bin = numBins - 1
			}
			binResiduals[bin] = append(binResiduals[bin], res)

			allValActuals = append(allValActuals, Y[i])
			allValPredictions = append(allValPredictions, pred)
		}
	}

	valN := float64(len(allValActuals))
	var valMAE, valRMSE, rSquared float64
	if valN > 0 {
		var sumY float64
		for _, y := range allValActuals {
			sumY += y
		}
		meanY := sumY / valN

		var maeSum, sse, sst float64
		for i := 0; i < len(allValActuals); i++ {
			diff := allValActuals[i] - allValPredictions[i]
			maeSum += math.Abs(diff)
			sse += diff * diff
			yDiff := allValActuals[i] - meanY
			sst += yDiff * yDiff
		}
		valMAE = maeSum / valN
		valRMSE = math.Sqrt(sse / valN)
		if sst > 1e-9 {
			rSquared = 1.0 - (sse / sst)
		} else {
			rSquared = 1.0
		}
	}

	metrics := CVMetrics{
		ValidationMAE:     valMAE,
		ValidationRMSE:    valRMSE,
		RSquared:          rSquared,
		ValidationSamples: len(allValActuals),
	}

	p10Offsets := make([]float64, numBins)
	p90Offsets := make([]float64, numBins)

	for b := 0; b < numBins; b++ {
		var rawP10, rawP90 float64
		if len(binResiduals[b]) > 0 {
			sort.Float64s(binResiduals[b])
			rawP10 = linearQuantile(binResiduals[b], 0.10)
			rawP90 = linearQuantile(binResiduals[b], 0.90)
		} else if b > 0 {
			rawP10 = p10Offsets[b-1]
			rawP90 = p90Offsets[b-1]
		}

		if b == 0 {
			p10Offsets[0] = math.Min(0.0, rawP10)
			p90Offsets[0] = math.Max(0.0, rawP90)
		} else {
			p10Offsets[b] = math.Min(p10Offsets[b-1], rawP10)
			p90Offsets[b] = math.Max(p90Offsets[b-1], rawP90)
		}
	}

	return HorizonOffsets{
		P10: p10Offsets,
		P90: p90Offsets,
	}, metrics, nil
}

// ChronologicalCV performs numFolds forward-chaining rolling-origin cross-validation on chronological data.
// In split k, it trains on folds 0..k-1 and validates on fold k using simulated forecast origin (timestamps[trainEnd-1]).
// It extracts empirical 10th and 90th percentile residuals per horizon bin, enforces monotonic expansion
// (uncertainty band widens as horizon increases), and returns the offsets and overall CV metrics.
func ChronologicalCV(
	X []FeatureVector,
	Y []float64,
	timestamps []time.Time,
	config EnsembleConfig,
	numFolds int,
	leadTimeFeatureIndex int, // Feature index where (slotTime - originTime)/24h is injected during validation, or -1 to skip
) (HorizonOffsets, CVMetrics, error) {
	return ChronologicalCVWithOptions(X, Y, timestamps, config, ChronologicalCVOptions{
		NumFolds:             numFolds,
		LeadTimeFeatureIndex: leadTimeFeatureIndex,
	})
}
