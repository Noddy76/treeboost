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

func TestHistogramSplit_ZeroUnderflowWeights(t *testing.T) {
	boundaries := []float64{0.0, 10.0, 20.0, 30.0}
	var hist FeatureHistogram

	// Bins with tiny or zero weights
	hist.Count[0] = 10
	hist.SumW[0] = 1e-12
	hist.SumWY[0] = 5e-11

	hist.Count[1] = 10
	hist.SumW[1] = 1e-12
	hist.SumWY[1] = 5e-11

	// Total weight near zero
	totalW := 2e-12
	totalWY := 1e-10

	_, _, ok := FindBestHistogramSplit(&hist, boundaries, totalW, totalWY)
	if ok {
		t.Errorf("expected no split when totalW is near zero underflow, got ok=true")
	}

	_, _, leftW, _, rightW, _, okC := FindBestHistogramSplitWithCounts(&hist, boundaries, totalW, totalWY, 20, 5)
	if okC {
		t.Errorf("expected no split for degenerate weights, got okC=true")
	}
	if leftW < 0.0 || rightW < 0.0 {
		t.Errorf("child weights must not be negative: leftW=%f, rightW=%f", leftW, rightW)
	}
}

func TestHistogramSplit_Over256Bins(t *testing.T) {
	boundaries := make([]float64, 300)
	for i := range boundaries {
		boundaries[i] = float64(i)
	}
	var hist FeatureHistogram
	for i := 0; i < 256; i++ {
		hist.Count[i] = 1
		hist.SumW[i] = 1.0
		hist.SumWY[i] = float64(i)
	}
	_, _, ok := FindBestHistogramSplit(&hist, boundaries, 256.0, 1000.0)
	if !ok {
		t.Errorf("expected split found")
	}
	_, _, _, _, _, _, okC := FindBestHistogramSplitWithCounts(&hist, boundaries, 256.0, 1000.0, 256, 1)
	if !okC {
		t.Errorf("expected split found with counts")
	}
}

func TestHistogramSplit_NonFiniteBoundaries(t *testing.T) {
	boundaries := []float64{math.Inf(1), math.Inf(1)}
	var hist FeatureHistogram
	hist.Count[0] = 10
	hist.SumW[0] = 5.0
	hist.SumWY[0] = 10.0
	hist.Count[1] = 10
	hist.SumW[1] = 5.0
	hist.SumWY[1] = 20.0
	splitVal, _, ok := FindBestHistogramSplit(&hist, boundaries, 10.0, 30.0)
	if ok {
		if math.IsInf(splitVal, 0) || math.IsNaN(splitVal) {
			t.Errorf("FindBestHistogramSplit accepted non-finite splitVal: %f", splitVal)
		}
	}
}

func TestHistogramSplit_ExtremeFiniteBoundariesMidpoint(t *testing.T) {
	// 1.5e308 and 1.7e308 are finite, but sum 3.2e308 exceeds MaxFloat64 (~1.79e308).
	// Midpoint 1.6e308 is finite and must be accepted without overflowing to +Inf.
	boundaries := []float64{1.5e308, 1.7e308}
	var hist FeatureHistogram
	hist.Count[0] = 10
	hist.SumW[0] = 5.0
	hist.SumWY[0] = 10.0
	hist.Count[1] = 10
	hist.SumW[1] = 5.0
	hist.SumWY[1] = 20.0

	splitVal, _, ok := FindBestHistogramSplit(&hist, boundaries, 10.0, 30.0)
	if !ok {
		t.Fatalf("expected FindBestHistogramSplit to find split for extreme finite boundaries")
	}
	if math.IsNaN(splitVal) || math.IsInf(splitVal, 0) {
		t.Fatalf("splitVal is non-finite: %f", splitVal)
	}
	expected := 1.6e308
	if math.Abs((splitVal-expected)/expected) > 1e-12 {
		t.Errorf("expected midpoint ~%e, got %e", expected, splitVal)
	}

	splitValC, _, _, _, _, _, okC := FindBestHistogramSplitWithCounts(&hist, boundaries, 10.0, 30.0, 20, 1)
	if !okC {
		t.Fatalf("expected FindBestHistogramSplitWithCounts to find split for extreme finite boundaries")
	}
	if math.IsNaN(splitValC) || math.IsInf(splitValC, 0) {
		t.Fatalf("splitValC is non-finite: %f", splitValC)
	}
	if math.Abs((splitValC-expected)/expected) > 1e-12 {
		t.Errorf("expected midpoint with counts ~%e, got %e", expected, splitValC)
	}
}

func TestHistogramSplit_WeightDiscrepancyNonNegativeRightWeight(t *testing.T) {
	boundaries := []float64{0.0, 10.0, 20.0}
	var hist FeatureHistogram

	// Simulate parent leaf.sumW (10.0) being less than actual child histogram bin sums (15.0)
	hist.Count[0] = 10
	hist.SumW[0] = 12.0
	hist.SumWY[0] = 24.0

	hist.Count[1] = 5
	hist.SumW[1] = 3.0
	hist.SumWY[1] = 6.0

	totalW := 10.0 // Mismatched smaller totalW from parent
	totalWY := 20.0

	// FindBestHistogramSplitWithCounts reconciles totalW from hist and clamps rightW to >= 0
	splitVal, _, leftW, leftWY, rightW, rightWY, ok := FindBestHistogramSplitWithCounts(&hist, boundaries, totalW, totalWY, 15, 1)
	if !ok {
		t.Fatalf("expected valid split, got ok=false")
	}
	if rightW < 0.0 {
		t.Errorf("rightW must never be negative, got %f", rightW)
	}
	if leftW <= 0.0 {
		t.Errorf("leftW must be positive, got %f", leftW)
	}
	if math.IsNaN(splitVal) || math.IsInf(splitVal, 0) {
		t.Errorf("splitVal is non-finite: %f", splitVal)
	}
	if math.IsNaN(leftWY) || math.IsNaN(rightWY) {
		t.Errorf("weighted target sums must be finite: leftWY=%f, rightWY=%f", leftWY, rightWY)
	}
}

// FuzzFindBestHistogramSplit fuzzes histogram split evaluation.
func FuzzFindBestHistogramSplit(f *testing.F) {
	// Seed 1: standard 3-bin histogram
	f.Add(float64(10.0), float64(20.0), float64(100.0), float64(5.0), float64(5.0), float64(50.0), 10, 1)
	// Seed 2: near-zero total weight
	f.Add(float64(0.0), float64(1.0), float64(1e-10), float64(1e-10), float64(1e-10), float64(1e-9), 5, 2)
	// Seed 3: negative totalWY and unequal bin weights
	f.Add(float64(-5.0), float64(5.0), float64(50.0), float64(-150.0), float64(25.0), float64(-75.0), 30, 5)

	f.Fuzz(func(t *testing.T, b0, b1, totalW, totalWY, w0, wy0 float64, count0, minSamples int) {
		var hist FeatureHistogram
		c := count0 % 100
		if c < 0 {
			c = -c
		}
		hist.Count[0] = c
		if !math.IsNaN(w0) && !math.IsInf(w0, 0) && w0 >= 0 {
			hist.SumW[0] = w0
		}
		if !math.IsNaN(wy0) && !math.IsInf(wy0, 0) {
			hist.SumWY[0] = wy0
		}

		// Also populate bin 1
		hist.Count[1] = c + 1
		hist.SumW[1] = 1.0
		hist.SumWY[1] = 2.0

		// Generate boundaries slice with varying lengths (from 2 up to 300)
		numBins := (count0 % 300)
		if numBins < 0 {
			numBins = -numBins
		}
		if numBins < 2 {
			numBins = 2
		}
		boundaries := make([]float64, numBins)
		boundaries[0] = b0
		step := math.Abs(b1-b0) + 1.0
		if math.IsNaN(step) || math.IsInf(step, 0) {
			step = 1.0
		}
		for i := 1; i < numBins; i++ {
			boundaries[i] = b0 + float64(i)*step
		}

		splitVal, score, ok := FindBestHistogramSplit(&hist, boundaries, totalW, totalWY)
		if ok {
			if math.IsNaN(splitVal) || math.IsInf(splitVal, 0) {
				t.Fatalf("FindBestHistogramSplit produced non-finite splitVal: %f", splitVal)
			}
			if math.IsNaN(score) || math.IsInf(score, 0) {
				t.Fatalf("FindBestHistogramSplit produced non-finite score: %f", score)
			}
		}

		sVal, sScore, leftW, leftWY, rightW, rightWY, okC := FindBestHistogramSplitWithCounts(
			&hist, boundaries, totalW, totalWY, c*2, minSamples)
		if okC {
			if math.IsNaN(sVal) || math.IsInf(sVal, 0) {
				t.Fatalf("FindBestHistogramSplitWithCounts produced non-finite splitVal: %f", sVal)
			}
			if math.IsNaN(sScore) || math.IsInf(sScore, 0) {
				t.Fatalf("FindBestHistogramSplitWithCounts produced non-finite score: %f", sScore)
			}
			if leftW < 0.0 || rightW < 0.0 {
				t.Fatalf("negative child weight: leftW=%f, rightW=%f", leftW, rightW)
			}
			if math.IsNaN(leftWY) || math.IsNaN(rightWY) {
				t.Fatalf("non-finite child WY: leftWY=%f, rightWY=%f", leftWY, rightWY)
			}
		}
	})
}

// FuzzHistogramPartitionParity tests the end-to-end parity between quantized histogram binning
// and continuous in-place partitioning under discrete, bounded, and degenerate features.
func FuzzHistogramPartitionParity(f *testing.F) {
	f.Add(float64(0.0), float64(1.0), float64(10.0), float64(50.0), float64(1.0), 42)
	f.Add(float64(-1.0), float64(1.0), float64(-100.0), float64(100.0), float64(1e-10), 17)
	f.Add(float64(5.0), float64(5.0), float64(68.38), float64(68.38), float64(100.0), 99)
	f.Add(float64(0.0), float64(6.0), float64(0.0), float64(1000.0), float64(1e-5), 7)

	f.Fuzz(func(t *testing.T, f0, f1, y0, y1, weightScale float64, patternSeed int) {
		if math.IsNaN(f0) || math.IsNaN(f1) || math.IsNaN(y0) || math.IsNaN(y1) || math.IsNaN(weightScale) {
			return
		}
		if math.IsInf(f0, 0) || math.IsInf(f1, 0) || math.IsInf(y0, 0) || math.IsInf(y1, 0) || math.IsInf(weightScale, 0) {
			return
		}

		const nSamples = 32
		const nFeatures = 2
		features := make([]float64, nFeatures*nSamples)
		targets := make([]float64, nSamples)
		weights := make([]float64, nSamples)

		wBase := math.Abs(weightScale)
		if wBase <= 0.0 || math.IsNaN(wBase) {
			wBase = 1.0
		}

		for i := 0; i < nSamples; i++ {
			// Create discrete/repeated pattern on feature 0 to challenge histogram boundaries
			mod := (i + patternSeed) % 4
			switch mod {
			case 0:
				features[i] = f0
			case 1:
				features[i] = f1
			case 2:
				features[i] = (f0 + f1) * 0.5
			default:
				features[i] = f0 + float64(i)*0.1
			}

			// Feature 1 is continuous
			features[nSamples+i] = float64(i) * 1.5

			// Targets
			if i%2 == 0 {
				targets[i] = y0
			} else {
				targets[i] = y1
			}

			// Weights
			weights[i] = wBase * (float64(i%3) + 0.1)
		}

		cd := ColumnarDataset{
			NumSamples:  nSamples,
			NumFeatures: nFeatures,
			Features:    features,
			Targets:     targets,
			Weights:     weights,
		}
		cd.Binned = QuantizeDataset(cd, 256)

		// 1. Direct check: Histogram slice vs In-place partition
		var hist FeatureHistogram
		sampleIdxs := make([]int, nSamples)
		for i := range sampleIdxs {
			sampleIdxs[i] = i
		}

		BuildHistogramSlice(cd.Binned.Bins, 0, cd.Weights, cd.Targets, sampleIdxs, 0, nSamples, &hist)
		splitVal, score, leftW, leftWY, rightW, rightWY, ok := FindBestHistogramSplitWithCounts(
			&hist, cd.Binned.BinBoundaries[0], 0, 0, nSamples, 2)

		if ok {
			if math.IsNaN(splitVal) || math.IsInf(splitVal, 0) {
				t.Fatalf("splitVal is non-finite: %f", splitVal)
			}
			if math.IsNaN(score) || math.IsInf(score, 0) {
				t.Fatalf("score is non-finite: %f", score)
			}
			if leftW < 0.0 || rightW < 0.0 {
				t.Fatalf("negative child weight: leftW=%f, rightW=%f", leftW, rightW)
			}
			if math.IsNaN(leftWY) || math.IsNaN(rightWY) {
				t.Fatalf("non-finite weighted sum: leftWY=%f, rightWY=%f", leftWY, rightWY)
			}

			// Verify in-place partitioning with this splitVal
			mid := partitionSamplesInPlace(sampleIdxs, 0, nSamples, cd.Features, 0, splitVal)
			if mid < 0 || mid > nSamples {
				t.Fatalf("partition mid %d out of bounds [0, %d]", mid, nSamples)
			}
		}

		// 2. Full Leafwise GBDT tree training on this challenging dataset
		const maxLeafBound = 500.0
		lgbParams := LightGBMParams{
			MaxDepth:        4,
			MaxLeaves:       8,
			MinChildSamples: 2,
			Estimators:      4,
			LearningRate:    0.1,
			L2LeafReg:       1.0,
			MaxLeafValue:    maxLeafBound,
		}

		trees, baseVal := TrainLeafwiseGBDTColumnar(cd, lgbParams)
		if math.IsNaN(baseVal) || math.IsInf(baseVal, 0) {
			t.Fatalf("TrainLeafwiseGBDTColumnar baseVal non-finite: %f", baseVal)
		}
		if math.Abs(baseVal) > maxLeafBound {
			t.Fatalf("TrainLeafwiseGBDTColumnar baseVal %f exceeds bound %f", baseVal, maxLeafBound)
		}

		for treeIdx, tree := range trees {
			for nodeIdx, node := range tree.Nodes {
				// Invariant A: No corrupted split node (must have both left and right children)
				if node.SplitFeature != -1 {
					if node.LeftChild < 0 || node.RightChild < 0 {
						t.Fatalf("tree %d node %d is corrupted: SplitFeature=%d but Left=%d, Right=%d",
							treeIdx, nodeIdx, node.SplitFeature, node.LeftChild, node.RightChild)
					}
				}
				// Invariant B: Leaf values must be finite and bounded
				if node.SplitFeature == -1 {
					if math.IsNaN(node.LeafValue) || math.IsInf(node.LeafValue, 0) {
						t.Fatalf("tree %d node %d has non-finite leaf value: %f", treeIdx, nodeIdx, node.LeafValue)
					}
					if math.Abs(node.LeafValue) > maxLeafBound {
						t.Fatalf("tree %d node %d leaf value %f exceeds bound %f",
							treeIdx, nodeIdx, node.LeafValue, maxLeafBound)
					}
				}
			}
		}

		// Invariant C: Evaluating tree produces finite predictions
		for i := 0; i < nSamples; i++ {
			pred := baseVal
			for _, tree := range trees {
				pred += lgbParams.LearningRate * EvaluateTreeColumnar(tree.Nodes, cd, i)
			}
			if math.IsNaN(pred) || math.IsInf(pred, 0) {
				t.Fatalf("prediction on sample %d is non-finite: %f", i, pred)
			}
		}
	})
}

// FuzzQuantizeDataset_SpecialFloats tests dataset quantization with non-finite and extreme float features.
func FuzzQuantizeDataset_SpecialFloats(f *testing.F) {
	f.Add(float64(0.0), float64(100.0), 16, 256)
	f.Add(float64(-1e5), float64(1e5), 8, 32)
	f.Add(float64(1.0), float64(1.0), 32, 64)
	f.Add(float64(1e308), float64(-1e308), 20, 128)

	f.Fuzz(func(t *testing.T, valA, valB float64, nSamples, maxBins int) {
		if nSamples < 2 {
			nSamples = 2
		}
		if nSamples > 64 {
			nSamples = 64
		}
		if maxBins < 2 {
			maxBins = 2
		}
		if maxBins > 256 {
			maxBins = 256
		}

		features := make([]float64, nSamples*2)
		for i := 0; i < nSamples; i++ {
			switch i % 5 {
			case 0:
				features[i] = valA
			case 1:
				features[i] = valB
			case 2:
				features[i] = math.NaN()
			case 3:
				features[i] = math.Inf(1)
			case 4:
				features[i] = math.Inf(-1)
			}
			features[nSamples+i] = float64(i)
		}

		cd := ColumnarDataset{
			NumSamples:  nSamples,
			NumFeatures: 2,
			Features:    features,
		}

		binned := QuantizeDataset(cd, maxBins)
		if binned.NumSamples != nSamples || binned.NumFeatures != 2 {
			t.Fatalf("unexpected dimensions: %d x %d", binned.NumSamples, binned.NumFeatures)
		}

		for fIdx := 0; fIdx < 2; fIdx++ {
			boundaries := binned.BinBoundaries[fIdx]
			if len(boundaries) < 1 {
				t.Fatalf("feature %d has empty boundaries", fIdx)
			}
			// Boundaries must be monotonic and finite
			for b := 0; b < len(boundaries); b++ {
				if math.IsNaN(boundaries[b]) || math.IsInf(boundaries[b], 0) {
					t.Fatalf("feature %d boundary %d is non-finite: %f", fIdx, b, boundaries[b])
				}
				if b > 0 && boundaries[b] <= boundaries[b-1] {
					t.Fatalf("feature %d boundaries non-monotonic: b[%d]=%f <= b[%d]=%f",
						fIdx, b, boundaries[b], b-1, boundaries[b-1])
				}
			}

			// Bins must be within [0, len(boundaries)-1]
			fOffset := fIdx * nSamples
			for i := 0; i < nSamples; i++ {
				binIdx := int(binned.Bins[fOffset+i])
				if binIdx < 0 || binIdx >= len(boundaries) {
					t.Fatalf("feature %d sample %d bin %d out of bounds [0, %d)",
						fIdx, i, binIdx, len(boundaries))
				}
			}
		}
	})
}
