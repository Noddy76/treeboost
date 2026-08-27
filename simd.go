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
	if hasAVX512 && len(indices) >= 32 {
		return computeWeightedSumAVX512(W, Y, indices)
	}
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
	if hasAVX512 && n >= 16 {
		return computeWeightedSumContiguousAVX512(W, Y)
	}
	if hasAVX2 && n >= 8 {
		return computeWeightedSumContiguousAVX2(W, Y)
	}
	return computeWeightedSumContiguousScalar(W, Y)
}
