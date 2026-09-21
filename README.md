# Treeboost: High-Performance GBDT & Multi-Algorithm Ensembles for Go

[![Go Reference](https://pkg.go.dev/badge/github.com/Noddy76/treeboost.svg)](https://pkg.go.dev/github.com/Noddy76/treeboost)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/Noddy76/treeboost/actions/workflows/ci.yml/badge.svg)](https://github.com/Noddy76/treeboost/actions)

`treeboost` is a fast, general-purpose machine learning library in Go for training, evaluating, and serving **Gradient Boosted Decision Trees (GBDT)** and multi-algorithm tree regression ensembles.

---

## ⚡ Highlights

- **Pure Go Core + AVX-512 / AVX2 SIMD**: Zero `cgo` dependencies. Includes handcrafted 512-bit AVX-512 and 256-bit AVX2 FMA assembly kernels (`weighted_sum_amd64.s` generated via [avo](https://github.com/mmcloughlin/avo)) with dynamic runtime CPU feature detection (`golang.org/x/sys/cpu`) and scalar fallbacks for ARM64 and non-AVX platforms.
  - **Contiguous Operations**: Uses 512-bit AVX-512 vectors ($N \ge 16$) when available, seamlessly scaling to 256-bit AVX2 ($N \ge 8$) or scalar.
  - **Indirect Indexing**: Uses tuned 256-bit AVX2 vectors for $16 \le N \le 2048$ and out-of-order scalar pipelining for large arrays, ensuring optimal cache and memory pipeline utilization on all CPU generations.
- **Multi-Algorithm Fusion**: Train and blend multiple distinct decision tree architectures within a single unified model:
  - **Symmetric Trees ([CatBoost](https://en.wikipedia.org/wiki/CatBoost))**: Obliviated decision trees where the same feature split condition is evaluated across an entire depth layer.
  - **Leaf-Wise Trees ([LightGBM](https://en.wikipedia.org/wiki/LightGBM))**: Best-first tree growth strategy splitting nodes with maximum loss reduction (gain) for rapid convergence.
  - **Extremely Randomized Trees ([ExtraTrees](https://en.wikipedia.org/wiki/Random_forest#ExtraTrees))**: Randomized feature cut-points that damp down variance and combat overfitting.
- **Histogram Quantization**: 256-bin equal-frequency feature quantization (`uint8` bin IDs) for $O(\text{numBins})$ histogram split finding.
- **Zero-Allocation Execution**: Pooled workspace buffers (`sync.Pool`), Structure-of-Arrays (`ColumnarDataset`), and in-place sample partitioning.
- **Vectorized Batch Inference**: `PredictBatch` evaluates multi-tree ensembles across sample batches with 0 heap allocations.
- **Serialization**: Native JSON model serialization (`SaveModel` / `LoadModel`).

---

## 📦 Installation

```bash
go get github.com/Noddy76/treeboost
```

---

## 🚀 Quick-Start Guide

### 1. Training a Multi-Algorithm Ensemble

```go
package main

import (
	"fmt"
	"log"

	"github.com/Noddy76/treeboost"
)

func main() {
	// 1. Prepare Training Features (X) and Targets (Y)
	X := []treeboost.FeatureVector{
		{Values: []float64{10.0, 50.0}},
		{Values: []float64{15.0, 30.0}},
		{Values: []float64{25.0, 10.0}},
		{Values: []float64{30.0, 5.0}},
	}
	Y := []float64{100.0, 80.0, 40.0, 20.0}

	// 2. Configure Ensemble Parameters
	config := treeboost.DefaultEnsembleConfig()
	config.CatBoost.Iterations = 100
	config.LightGBM.Estimators = 100
	config.ExtraTrees.Estimators = 100

	// 3. Train Model
	model := treeboost.TrainEnsemble(X, Y, config)
	fmt.Printf("Model trained! R^2: %.4f, MAE: %.4f, RMSE: %.4f\n", model.RSquared, model.MAE, model.RMSE)

	// 4. Predict Single Observation
	pred := model.Predict([]float64{20.0, 20.0})
	fmt.Printf("Predicted Value: %.2f\n", pred)

	// 5. Predict Batch (Zero-Alloc Vectorized)
	flatFeatures := []float64{
		10.0, 50.0,
		25.0, 10.0,
	}
	batchPreds := model.PredictBatch(flatFeatures, 2)
	fmt.Printf("Batch Predictions: %v\n", batchPreds)

	// 6. Save & Load Model
	if err := model.SaveModel("model.json"); err != nil {
		log.Fatal(err)
	}
	loadedModel, err := treeboost.LoadModel("model.json")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Loaded model prediction: %.2f\n", loadedModel.Predict([]float64{20.0, 20.0}))
}
```

### 2. Training Individual GBDT Algorithms

You can also train individual tree algorithms standalone:

```go
W := treeboost.ComputeSampleWeights(Y)

// CatBoost (Symmetric Obliviated Trees)
cbTrees, cbBase := treeboost.TrainSymmetricGBDT(X, Y, W, treeboost.CatBoostParams{
	Depth:        6,
	Iterations:   200,
	LearningRate: 0.05,
	L2LeafReg:    3.0,
})

// LightGBM (Leaf-Wise Best-First Trees)
lgbTrees, lgbBase := treeboost.TrainLeafwiseGBDT(X, Y, W, treeboost.LightGBMParams{
	MaxDepth:        6,
	MaxLeaves:       31,
	MinChildSamples: 20,
	Estimators:      200,
	LearningRate:    0.05,
	L2LeafReg:       1.0,
})

// ExtraTrees (Extremely Randomized Trees)
etTrees := treeboost.TrainExtraTrees(X, Y, W, treeboost.ExtraTreesParams{
	Estimators:     300,
	MinSamplesLeaf: 4,
	MaxFeatures:    1.0,
})
```

---

## 📊 Benchmarks

Run benchmarks locally:

```bash
make bench
# or: go test -v -bench=. -benchmem ./...
```

### 1. Vectorized Memory Bandwidth & Latency ($N=1,024$)

| Kernel | Scalar (No SIMD) | AVX2 (256-bit) | AVX-512 (512-bit) | AVX-512 vs Scalar | AVX-512 vs AVX2 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`WeightedSumContiguous`** | $345.9\text{ ns}$ ($47\text{ GB/s}$) | $162.9\text{ ns}$ ($100\text{ GB/s}$) | **$95.5\text{ ns}$ ($171.6\text{ GB/s}$)** | **$3.62\times$ faster** | **$1.71\times$ faster** |
| **`WeightedSumIndirect`** ($N=256$) | $219.3\text{ ns}$ | **$101.4\text{ ns}$** | **$100.6\text{ ns}$** | **$2.18\times$ faster** | $1.01\times$ |
| **`WeightedSumIndirect`** ($N=1,024$) | $923.7\text{ ns}$ | **$381.3\text{ ns}$** | **$377.4\text{ ns}$** | **$2.45\times$ faster** | $1.01\times$ |

### 2. End-to-End Ensemble Training & Inference

| Benchmark | Latency | Memory / Allocs | Notes |
| :--- | :--- | :--- | :--- |
| **`BenchmarkGBDTPredict`** (Single Sample) | **2.56 µs/op** | 0 B/op, 0 allocs | Real-time serving latency |
| **`BenchmarkPredictBatch`** (64 Samples) | **245 µs/op** | 512 B/op, 1 alloc | Multi-sample batch scoring |
| **`BenchmarkGBDTTrainEnsemble`** (AVX-512) | **71.5 ms/op** | 2.93 MB/op, 859 allocs | Full CatBoost + LightGBM + ExtraTrees |
| **`BenchmarkGBDTTrainEnsemble`** (AVX2-Only) | **72.3 ms/op** | 2.93 MB/op, 859 allocs | Automatic seamless fallback |

> **Dynamic SIMD Dispatch**: Treeboost dynamically detects CPU flags at startup (`HasAVX512`, `HasAVX2`, `HasFMA`). Contiguous data paths maximize 512-bit vector throughput, while indirect index paths leverage tuned 256-bit AVX2 kernels and out-of-order execution, ensuring top performance across all Intel and AMD x86-64 hardware.

---

## 🛠️ Regenerating SIMD Assembly

The pre-generated assembly file `weighted_sum_amd64.s` is included directly in the repository so no special tools are needed for standard builds. If you modify the assembly generator, regenerate it using:

```bash
go generate ./...
# or: go run ./cmd/gen_simd -out weighted_sum_amd64.s
```

---

## 📚 Theoretical Background & References

1. **Gradient Boosting**: [Wikipedia - Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) — Numerical optimization framework iteratively fitting weak trees to negative loss gradients.
2. **Symmetric Decision Trees (CatBoost)**: [Wikipedia - CatBoost](https://en.wikipedia.org/wiki/CatBoost) — Obliviated trees evaluating identical split conditions across all nodes at depth $d$.
3. **Leaf-wise Decision Trees (LightGBM)**: [Wikipedia - LightGBM](https://en.wikipedia.org/wiki/LightGBM) — Best-first tree expansion strategy splitting nodes with maximum loss reduction.
4. **Extremely Randomized Trees (ExtraTrees)**: [Wikipedia - Random Forest / ExtraTrees](https://en.wikipedia.org/wiki/Random_forest#ExtraTrees) — Randomizing feature thresholds to reduce variance.

---

## 📄 License

Licensed under the [Apache License, Version 2.0](LICENSE).
