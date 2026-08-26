//go:build amd64

//go:generate go run ./cmd/gen_simd -out weighted_sum_amd64.s

package treeboost

// computeWeightedSumAVX2 is implemented in weighted_sum_amd64.s
func computeWeightedSumAVX2(W []float64, Y []float64, indices []int) (sumW float64, sumWY float64)

// computeWeightedSumContiguousAVX2 is implemented in weighted_sum_amd64.s
func computeWeightedSumContiguousAVX2(W []float64, Y []float64) (sumW float64, sumWY float64)
