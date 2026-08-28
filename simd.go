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

import "golang.org/x/sys/cpu"

var (
	hasAVX2   = cpu.X86.HasAVX2 && cpu.X86.HasFMA
	hasAVX512 = cpu.X86.HasAVX512 && cpu.X86.HasAVX512F && cpu.X86.HasAVX512DQ
)

func computeWeightedSumScalar(W []float64, Y []float64, indices []int) (float64, float64) {
	var sumW, sumWY float64
	for _, idx := range indices {
		w := W[idx]
		sumW += w
		sumWY += w * Y[idx]
	}
	return sumW, sumWY
}

func computeWeightedSum(W []float64, Y []float64, indices []int) (float64, float64) {
	if hasAVX2 && len(indices) >= 16 && len(indices) <= 2048 {
		return computeWeightedSumAVX2(W, Y, indices)
	}
	return computeWeightedSumScalar(W, Y, indices)
}

func computeWeightedSumContiguousScalar(W []float64, Y []float64) (float64, float64) {
	n := len(W)
	if len(Y) < n {
		n = len(Y)
	}
	var sumW, sumWY float64
	for i := 0; i < n; i++ {
		w := W[i]
		sumW += w
		sumWY += w * Y[i]
	}
	return sumW, sumWY
}

func computeWeightedSumContiguous(W []float64, Y []float64) (float64, float64) {
	n := len(W)
	if len(Y) < n {
		n = len(Y)
	}
	if hasAVX512 && n >= 16 {
		return computeWeightedSumContiguousAVX512(W, Y)
	}
	if hasAVX2 && n >= 8 {
		return computeWeightedSumContiguousAVX2(W, Y)
	}
	return computeWeightedSumContiguousScalar(W, Y)
}
