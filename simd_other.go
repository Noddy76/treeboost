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

//go:build !amd64

package treeboost

func computeWeightedSumAVX512(W []float64, Y []float64, indices []int) (sumW float64, sumWY float64) {
	return computeWeightedSumScalar(W, Y, indices)
}

func computeWeightedSumContiguousAVX512(W []float64, Y []float64) (sumW float64, sumWY float64) {
	return computeWeightedSumContiguousScalar(W, Y)
}

func computeWeightedSumAVX2(W []float64, Y []float64, indices []int) (sumW float64, sumWY float64) {
	return computeWeightedSumScalar(W, Y, indices)
}

func computeWeightedSumContiguousAVX2(W []float64, Y []float64) (sumW float64, sumWY float64) {
	return computeWeightedSumContiguousScalar(W, Y)
}
