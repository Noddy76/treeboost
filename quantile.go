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
	"slices"
)

// Supported loss function identifier constants.
const (
	LossMSE     = "mse"
	LossPinball = "pinball"
)

// validateQuantileParams validates the loss function and target quantile.
// If lossFunction is "pinball", quantile must be finite and strictly within (0.0, 1.0).
// If invalid or unset, it panics immediately with a descriptive message.
func validateQuantileParams(lossFunction string, quantile float64) {
	if lossFunction == LossPinball {
		if math.IsNaN(quantile) || math.IsInf(quantile, 0) || quantile <= 0.0 || quantile >= 1.0 {
			panic(fmt.Sprintf("treeboost: invalid quantile %v; must be finite and in (0.0, 1.0)", quantile))
		}
	}
}

// isPinballLoss returns true if lossFunction is LossPinball.
func isPinballLoss(lossFunction string) bool {
	return lossFunction == LossPinball
}

// WeightedQuantile computes the weighted q-quantile of values with corresponding sample weights.
// If weights is nil or empty, all weights default to 1.0 (uniform weighting).
// Target quantile q is clamped to [0.0, 1.0]. If q is NaN, it returns 0.0.
// Non-finite values (NaN, +/-Inf) and non-positive or non-finite weights are safely filtered out.
func WeightedQuantile(values []float64, weights []float64, q float64) float64 {
	if len(values) == 0 {
		return 0.0
	}
	if len(weights) == 0 {
		weights = nil
	} else if len(weights) != len(values) {
		return 0.0
	}
	sampleIdxs := make([]int, len(values))
	for i := range sampleIdxs {
		sampleIdxs[i] = i
	}
	scratch := make([]int, 0, len(values))
	return weightedQuantileIndexed(sampleIdxs, values, weights, q, &scratch)
}

// weightedQuantileIndexed computes the weighted q-quantile of values[sampleIdxs] using scratch buffer
// to eliminate heap allocations inside boosting loops.
func weightedQuantileIndexed(sampleIdxs []int, values []float64, weights []float64, q float64, scratch *[]int) float64 {
	if len(sampleIdxs) == 0 || len(values) == 0 {
		return 0.0
	}
	if len(weights) == 0 {
		weights = nil
	}
	if math.IsNaN(q) {
		return 0.0
	}
	if q < 0.0 {
		q = 0.0
	} else if q > 1.0 {
		q = 1.0
	}

	var valid []int
	if scratch != nil {
		if cap(*scratch) < len(sampleIdxs) {
			*scratch = make([]int, 0, len(sampleIdxs))
		}
		valid = (*scratch)[:0]
	} else {
		valid = make([]int, 0, len(sampleIdxs))
	}

	for _, idx := range sampleIdxs {
		if idx < 0 || idx >= len(values) {
			continue
		}
		v := values[idx]
		if math.IsNaN(v) || math.IsInf(v, 0) {
			continue
		}
		if weights != nil {
			if idx >= len(weights) {
				continue
			}
			w := weights[idx]
			if w <= 0.0 || math.IsNaN(w) || math.IsInf(w, 0) {
				continue
			}
		}
		valid = append(valid, idx)
	}

	if scratch != nil {
		*scratch = valid
	}

	m := len(valid)
	if m == 0 {
		return 0.0
	}
	if m == 1 {
		return values[valid[0]]
	}

	slices.SortFunc(valid, func(a, b int) int {
		va, vb := values[a], values[b]
		if va < vb {
			return -1
		}
		if va > vb {
			return 1
		}
		return 0
	})

	if values[valid[0]] == values[valid[m-1]] {
		return values[valid[0]]
	}

	var totalW float64
	for _, idx := range valid {
		w := 1.0
		if weights != nil && idx < len(weights) {
			w = weights[idx]
		}
		totalW += w
	}

	wScale := 1.0
	if totalW <= 0.0 || math.IsNaN(totalW) || math.IsInf(totalW, 0) {
		// Attempt rescaling by maxW if weights underflowed to 0 or overflowed to Inf
		maxW := 0.0
		for _, idx := range valid {
			if weights != nil && idx < len(weights) {
				w := weights[idx]
				if w > maxW && !math.IsNaN(w) && !math.IsInf(w, 0) {
					maxW = w
				}
			}
		}
		if maxW > 0.0 {
			wScale = 1.0 / maxW
			totalW = 0.0
			for _, idx := range valid {
				w := 1.0
				if weights != nil && idx < len(weights) {
					w = weights[idx] * wScale
				}
				totalW += w
			}
		}
	}

	if totalW <= 0.0 || math.IsNaN(totalW) || math.IsInf(totalW, 0) {
		return 0.0
	}

	getW := func(k int) float64 {
		if weights != nil && valid[k] < len(weights) {
			return weights[valid[k]] * wScale
		}
		return 1.0 * wScale
	}

	w0 := getW(0)
	c0 := 0.5 * w0
	wm1 := getW(m - 1)
	cLast := totalW - 0.5*wm1

	T := q * totalW
	if T <= c0 {
		return values[valid[0]]
	}
	if T >= cLast {
		return values[valid[m-1]]
	}

	currC := c0
	currW := w0
	for k := 0; k < m-1; k++ {
		nextW := getW(k + 1)
		nextC := currC + 0.5*currW + 0.5*nextW
		if T <= nextC {
			denom := nextC - currC
			if denom <= 0.0 || math.IsNaN(denom) {
				return values[valid[k]]
			}
			t := (T - currC) / denom
			if t < 0.0 {
				t = 0.0
			} else if t > 1.0 {
				t = 1.0
			}
			vk := values[valid[k]]
			vk1 := values[valid[k+1]]
			diff := vk1 - vk
			if math.IsInf(diff, 0) {
				res := (1.0-t)*vk + t*vk1
				if math.IsNaN(res) || math.IsInf(res, 0) {
					return vk
				}
				return res
			}
			return vk + t*diff
		}
		currC = nextC
		currW = nextW
	}

	return values[valid[m-1]]
}

// PinballLoss evaluates the weighted mean pinball (quantile/check) loss:
//
//	L_q(y, y_hat) = w * max(q * (y - y_hat), (q - 1) * (y - y_hat))
//
// Returns 0.0 if datasets are empty, length mismatched, or total valid weight is non-positive.
func PinballLoss(yTrue, yPred, weights []float64, q float64) float64 {
	if len(yTrue) == 0 || len(yPred) == 0 || len(yTrue) != len(yPred) {
		return 0.0
	}
	if len(weights) == 0 {
		weights = nil
	} else if len(weights) != len(yTrue) {
		return 0.0
	}
	if math.IsNaN(q) || math.IsInf(q, 0) {
		return 0.0
	}
	if q < 0.0 {
		q = 0.0
	} else if q > 1.0 {
		q = 1.0
	}

	var totalW, totalLoss float64
	for i := 0; i < len(yTrue); i++ {
		yt := yTrue[i]
		yp := yPred[i]
		if math.IsNaN(yt) || math.IsInf(yt, 0) || math.IsNaN(yp) || math.IsInf(yp, 0) {
			continue
		}
		diff := yt - yp
		if math.IsNaN(diff) || math.IsInf(diff, 0) {
			continue
		}
		w := 1.0
		if weights != nil {
			if i >= len(weights) {
				continue
			}
			w = weights[i]
			if w <= 0.0 || math.IsNaN(w) || math.IsInf(w, 0) {
				continue
			}
		}

		var loss float64
		if diff > 0 {
			loss = q * diff
		} else {
			loss = (q - 1.0) * diff
		}
		if math.IsNaN(loss) || math.IsInf(loss, 0) || loss < 0.0 {
			continue
		}
		wl := w * loss
		if math.IsNaN(wl) || math.IsInf(wl, 0) || wl < 0.0 {
			continue
		}

		totalLoss += wl
		totalW += w
	}

	if totalW <= 0.0 || math.IsNaN(totalW) || math.IsInf(totalW, 0) {
		return 0.0
	}
	res := totalLoss / totalW
	if math.IsNaN(res) || math.IsInf(res, 0) || res < 0.0 {
		return 0.0
	}
	return res
}
