package treeboost

import "golang.org/x/sys/cpu"

var hasAVX2 = cpu.X86.HasAVX2 && cpu.X86.HasFMA

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
	if hasAVX2 && len(indices) >= 16 {
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
	if hasAVX2 && n >= 8 {
		return computeWeightedSumContiguousAVX2(W, Y)
	}
	return computeWeightedSumContiguousScalar(W, Y)
}
