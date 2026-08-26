//go:build !amd64

package treeboost

func computeWeightedSumAVX2(W []float64, Y []float64, indices []int) (sumW float64, sumWY float64) {
	return computeWeightedSumScalar(W, Y, indices)
}

func computeWeightedSumContiguousAVX2(W []float64, Y []float64) (sumW float64, sumWY float64) {
	return computeWeightedSumContiguousScalar(W, Y)
}
