package treeboost

import (
	"fmt"
	"testing"
)

func BenchmarkSizesContiguous(b *testing.B) {
	sizes := []int{16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536}
	for _, n := range sizes {
		W := make([]float64, n)
		Y := make([]float64, n)
		for i := 0; i < n; i++ {
			W[i] = float64(i+1) * 0.5
			Y[i] = float64((i*17)%31) - 15.0
		}

		b.Run(fmt.Sprintf("Size=%d/AVX512", n), func(b *testing.B) {
			if !hasAVX512 {
				b.Skip("no AVX-512")
			}
			b.SetBytes(int64(n * 16)) // 2 * float64 = 16 bytes per element
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = computeWeightedSumContiguousAVX512(W, Y)
			}
		})

		b.Run(fmt.Sprintf("Size=%d/AVX2", n), func(b *testing.B) {
			if !hasAVX2 {
				b.Skip("no AVX2")
			}
			b.SetBytes(int64(n * 16))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = computeWeightedSumContiguousAVX2(W, Y)
			}
		})

		b.Run(fmt.Sprintf("Size=%d/Scalar", n), func(b *testing.B) {
			b.SetBytes(int64(n * 16))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = computeWeightedSumContiguousScalar(W, Y)
			}
		})
	}
}

func BenchmarkSizesIndirect(b *testing.B) {
	sizes := []int{16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536}
	for _, n := range sizes {
		W := make([]float64, n)
		Y := make([]float64, n)
		indices := make([]int, n)
		for i := 0; i < n; i++ {
			W[i] = float64(i+1) * 0.5
			Y[i] = float64((i*17)%31) - 15.0
			indices[i] = i
		}

		b.Run(fmt.Sprintf("Size=%d/AVX512", n), func(b *testing.B) {
			if !hasAVX512 {
				b.Skip("no AVX-512")
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = computeWeightedSumAVX512(W, Y, indices)
			}
		})

		b.Run(fmt.Sprintf("Size=%d/AVX2", n), func(b *testing.B) {
			if !hasAVX2 {
				b.Skip("no AVX2")
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = computeWeightedSumAVX2(W, Y, indices)
			}
		})

		b.Run(fmt.Sprintf("Size=%d/Scalar", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = computeWeightedSumScalar(W, Y, indices)
			}
		})
	}
}

func BenchmarkGBDTTrainingModes(b *testing.B) {
	nSamples := 1000
	nFeatures := 10
	X := make([]FeatureVector, nSamples)
	Y := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		vals := make([]float64, nFeatures)
		for f := 0; f < nFeatures; f++ {
			vals[f] = float64((i*37 + f*13) % 100)
		}
		X[i] = FeatureVector{Values: vals}
		Y[i] = vals[0]*2.0 + vals[1]*0.5
	}

	config := EnsembleConfig{
		CatBoostWeight:   0.4,
		LightGBMWeight:   0.4,
		ExtraTreesWeight: 0.2,
		CatBoost: CatBoostParams{
			Depth:        4,
			Iterations:   100,
			LearningRate: 0.1,
			L2LeafReg:    1.0,
		},
		LightGBM: LightGBMParams{
			MaxDepth:        4,
			MaxLeaves:       16,
			MinChildSamples: 1,
			Estimators:      100,
			LearningRate:    0.1,
		},
		ExtraTrees: ExtraTreesParams{
			Estimators:     100,
			MinSamplesLeaf: 1,
			MaxFeatures:    1.0,
		},
	}

	// Backup feature flags
	origAVX512 := hasAVX512
	origAVX2 := hasAVX2
	defer func() {
		hasAVX512 = origAVX512
		hasAVX2 = origAVX2
	}()

	b.Run("Mode=AVX512", func(b *testing.B) {
		if !origAVX512 {
			b.Skip("AVX-512 not supported")
		}
		hasAVX512 = true
		hasAVX2 = true
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = TrainEnsemble(X, Y, config)
		}
	})

	b.Run("Mode=AVX2_Only", func(b *testing.B) {
		if !origAVX2 {
			b.Skip("AVX2 not supported")
		}
		hasAVX512 = false
		hasAVX2 = true
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = TrainEnsemble(X, Y, config)
		}
	})

	b.Run("Mode=Scalar_Only", func(b *testing.B) {
		hasAVX512 = false
		hasAVX2 = false
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = TrainEnsemble(X, Y, config)
		}
	})
}

func BenchmarkDispatchedContiguous(b *testing.B) {
	origAVX512 := hasAVX512
	origAVX2 := hasAVX2
	defer func() {
		hasAVX512 = origAVX512
		hasAVX2 = origAVX2
	}()

	n := 1024
	W := make([]float64, n)
	Y := make([]float64, n)
	for i := 0; i < n; i++ {
		W[i] = float64(i+1) * 0.5
		Y[i] = float64((i*17)%31) - 15.0
	}

	modes := []struct {
		name   string
		avx512 bool
		avx2   bool
	}{
		{"AVX512", true, true},
		{"AVX2_Only", false, true},
		{"Scalar_Only", false, false},
	}

	for _, m := range modes {
		b.Run(m.name, func(b *testing.B) {
			if m.avx512 && !origAVX512 {
				b.Skip("Host does not support AVX-512")
			}
			if m.avx2 && !origAVX2 {
				b.Skip("Host does not support AVX2")
			}
			hasAVX512 = m.avx512
			hasAVX2 = m.avx2
			b.SetBytes(int64(n * 16))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = computeWeightedSumContiguous(W, Y)
			}
		})
	}
}

func BenchmarkDispatchedIndirect(b *testing.B) {
	origAVX512 := hasAVX512
	origAVX2 := hasAVX2
	defer func() {
		hasAVX512 = origAVX512
		hasAVX2 = origAVX2
	}()

	sizes := []int{64, 256, 1024, 4096}
	modes := []struct {
		name   string
		avx512 bool
		avx2   bool
	}{
		{"AVX512_System", true, true},
		{"AVX2_Only_System", false, true},
		{"Scalar_Only_System", false, false},
	}

	for _, n := range sizes {
		W := make([]float64, n)
		Y := make([]float64, n)
		indices := make([]int, n)
		for i := 0; i < n; i++ {
			W[i] = float64(i+1) * 0.5
			Y[i] = float64((i*17)%31) - 15.0
			indices[i] = i
		}

		for _, m := range modes {
			b.Run(fmt.Sprintf("Size=%d/%s", n, m.name), func(b *testing.B) {
				if m.avx512 && !origAVX512 {
					b.Skip("Host does not support AVX-512")
				}
				if m.avx2 && !origAVX2 {
					b.Skip("Host does not support AVX2")
				}
				hasAVX512 = m.avx512
				hasAVX2 = m.avx2
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = computeWeightedSum(W, Y, indices)
				}
			})
		}
	}
}
