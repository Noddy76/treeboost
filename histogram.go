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
	"sort"
)

// BinnedDataset represents a tabular dataset whose continuous features are quantized
// into discrete 8-bit bin IDs (0..maxBins-1) for fast histogram-based GBDT tree building.
type BinnedDataset struct {
	NumSamples    int
	NumFeatures   int
	MaxBins       int
	Bins          []uint8     // Size (NumFeatures * NumSamples)
	BinBoundaries [][]float64 // Bin upper bound thresholds per feature [NumFeatures][]float64
	BinCounts     []int       // Actual number of bins created per feature [NumFeatures]int
}

// FeatureHistogram holds accumulated count, weight, and weighted target sums across 256 discrete bins.
type FeatureHistogram struct {
	Count [256]int
	SumW  [256]float64
	SumWY [256]float64
}

// Reset zeroes out histogram accumulators.
func (h *FeatureHistogram) Reset() {
	for b := 0; b < 256; b++ {
		h.Count[b] = 0
		h.SumW[b] = 0.0
		h.SumWY[b] = 0.0
	}
}

// QuantizeDataset builds a BinnedDataset from a ColumnarDataset using equal-frequency quantile binning.
func QuantizeDataset(data ColumnarDataset, maxBins int) BinnedDataset {
	if maxBins <= 0 || maxBins > 256 {
		maxBins = 256
	}
	if data.NumSamples == 0 || data.NumFeatures == 0 {
		return BinnedDataset{
			NumSamples:  data.NumSamples,
			NumFeatures: data.NumFeatures,
			MaxBins:     maxBins,
		}
	}

	nSamples := data.NumSamples
	nFeatures := data.NumFeatures
	bins := make([]uint8, nFeatures*nSamples)
	binBoundaries := make([][]float64, nFeatures)
	binCounts := make([]int, nFeatures)

	scratch := make([]float64, nSamples)

	for f := 0; f < nFeatures; f++ {
		fOffset := f * nSamples
		copy(scratch, data.Features[fOffset:fOffset+nSamples])
		sort.Float64s(scratch)

		var boundaries []float64
		step := float64(nSamples) / float64(maxBins)
		if step < 1.0 {
			step = 1.0
		}
		lastVal := scratch[0]
		boundaries = append(boundaries, lastVal)

		for b := 1; b < maxBins; b++ {
			idx := int(float64(b) * step)
			if idx >= nSamples {
				idx = nSamples - 1
			}
			val := scratch[idx]
			if val > lastVal {
				boundaries = append(boundaries, val)
				lastVal = val
			}
		}

		numBins := len(boundaries)
		binCounts[f] = numBins
		binBoundaries[f] = boundaries

		for i := 0; i < nSamples; i++ {
			val := data.Features[fOffset+i]
			binIdx := sort.SearchFloat64s(boundaries, val)
			if binIdx >= numBins {
				binIdx = numBins - 1
			}
			bins[fOffset+i] = uint8(binIdx)
		}
	}

	return BinnedDataset{
		NumSamples:    nSamples,
		NumFeatures:   nFeatures,
		MaxBins:       maxBins,
		Bins:          bins,
		BinBoundaries: binBoundaries,
		BinCounts:     binCounts,
	}
}

// BuildHistogram accumulates sample weights and targets into a FeatureHistogram.
func BuildHistogram(bins []uint8, fOffset int, weights []float64, targets []float64, indices []int, hist *FeatureHistogram) {
	hist.Reset()
	for _, idx := range indices {
		b := bins[fOffset+idx]
		w := weights[idx]
		hist.Count[b]++
		hist.SumW[b] += w
		hist.SumWY[b] += w * targets[idx]
	}
}

// BuildHistogramSlice accumulates sample weights and targets over sampleIdxs[start:end] without slice allocations.
func BuildHistogramSlice(bins []uint8, fOffset int, weights []float64, targets []float64, sampleIdxs []int, start, end int, hist *FeatureHistogram) {
	hist.Reset()
	for i := start; i < end; i++ {
		idx := sampleIdxs[i]
		b := bins[fOffset+idx]
		w := weights[idx]
		hist.Count[b]++
		hist.SumW[b] += w
		hist.SumWY[b] += w * targets[idx]
	}
}

// FindBestHistogramSplit scans accumulated feature histogram bins to find the split threshold
// minimizing variance (maximizing gain).
func FindBestHistogramSplit(hist *FeatureHistogram, boundaries []float64, totalW, totalWY float64) (float64, float64, bool) {
	numBins := len(boundaries)
	if numBins <= 1 || totalW <= 1e-9 {
		return 0.0, math.MaxFloat64, false
	}

	var leftW, leftWY float64
	bestScore := math.MaxFloat64
	bestSplitVal := 0.0
	found := false

	for b := 0; b < numBins-1; b++ {
		leftW += hist.SumW[b]
		leftWY += hist.SumWY[b]

		rightW := totalW - leftW
		rightWY := totalWY - leftWY

		if leftW <= 1e-9 || rightW <= 1e-9 {
			continue
		}

		score := -((leftWY * leftWY / leftW) + (rightWY * rightWY / rightW))

		if score < bestScore {
			bestScore = score
			if b+1 < numBins {
				bestSplitVal = (boundaries[b] + boundaries[b+1]) / 2.0
			} else {
				bestSplitVal = boundaries[b]
			}
			found = true
		}
	}

	return bestSplitVal, bestScore, found
}

// FindBestHistogramSplitWithCounts evaluates the optimal split threshold and partition metrics directly from histogram bins.
func FindBestHistogramSplitWithCounts(hist *FeatureHistogram, boundaries []float64, totalW, totalWY float64, totalCount int, minChildSamples int) (splitVal float64, score float64, leftW float64, leftWY float64, rightW float64, rightWY float64, found bool) {
	numBins := len(boundaries)
	if numBins <= 1 || totalW <= 1e-9 {
		return 0.0, math.MaxFloat64, 0, 0, 0, 0, false
	}

	var curLeftW, curLeftWY float64
	var curLeftCount int
	bestScore := math.MaxFloat64
	bestSplitVal := 0.0
	var bestLeftW, bestLeftWY, bestRightW, bestRightWY float64

	for b := 0; b < numBins-1; b++ {
		curLeftCount += hist.Count[b]
		curRightCount := totalCount - curLeftCount
		curLeftW += hist.SumW[b]
		curLeftWY += hist.SumWY[b]

		if curLeftCount < minChildSamples || curRightCount < minChildSamples {
			continue
		}

		curRightW := totalW - curLeftW
		curRightWY := totalWY - curLeftWY

		if curLeftW <= 1e-9 || curRightW <= 1e-9 {
			continue
		}

		scoreVal := -((curLeftWY * curLeftWY / curLeftW) + (curRightWY * curRightWY / curRightW))

		if scoreVal < bestScore {
			bestScore = scoreVal
			if b+1 < numBins {
				bestSplitVal = (boundaries[b] + boundaries[b+1]) / 2.0
			} else {
				bestSplitVal = boundaries[b]
			}
			bestLeftW = curLeftW
			bestLeftWY = curLeftWY
			bestRightW = curRightW
			bestRightWY = curRightWY
			found = true
		}
	}

	return bestSplitVal, bestScore, bestLeftW, bestLeftWY, bestRightW, bestRightWY, found
}
