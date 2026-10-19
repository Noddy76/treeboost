# Release v0.7.0: Histogram L2 Regularization, Multi-Node Level Histograms, Small-N Concurrency Guard, and Pre-Binned Columnar Entry Points

## Overview & Motivation
`treeboost` v0.7.0 delivers foundational algorithmic and throughput optimizations for high-dimensional and walk-forward GBDT training pipelines (implementing Phase A of `TRAINING_ACCELERATION_PLAN.md`). It addresses split quality on small-sample outlier bins, eliminates raw `float64` scans in CatBoost-style symmetric trees, guards against goroutine/pool thrashing when training under outer model parallelism, and exposes pre-binned `ColumnarDataset` entry points that allow concurrent quantile learners to share quantized feature matrices with zero redundant work.

## Key Capabilities & Algorithmic Implementations

### 1. L2-Regularized Histogram Split Scoring (`FindBestHistogramSplitReg`, `FindBestHistogramSplitWithCountsReg`)
- Introduced L2 regularization ($\lambda \ge 0$) to histogram bin evaluation:
  $$\text{Score}(b) = -\left(\frac{(\sum_{k \le b} W_k Y_k)^2}{\sum_{k \le b} W_k + \lambda} + \frac{(\sum_{k > b} W_k Y_k)^2}{\sum_{k > b} W_k + \lambda}\right)$$
- Prevents low-weight outlier bins from artificially dominating split selection.
- Fail-fast parameter validation: panics immediately on negative, NaN, or non-finite $\lambda$.
- Legacy signatures `FindBestHistogramSplit` and `FindBestHistogramSplitWithCounts` remain wrappers defaulting to $\lambda = 0.0$ for 100% backward compatibility.
- LightGBM-style leafwise split finding (`evaluateLeafwiseFeatureSplits`) now passes `params.L2LeafReg` into histogram scoring.

### 2. Multi-Node Level Histograms for Symmetric Trees
- Level-wise split search in CatBoost-style symmetric trees now evaluates split gain summed across all $2^d$ active nodes at depth $d$, rather than relying on a global unpartitioned root histogram.
- Evaluates histogram prefix sums using fixed `ws.LevelScore [256]float64` and `ws.LevelHasSplit [256]bool` workspace buffers, eliminating heap allocations ($0\text{ allocs/op}$) while keeping memory bounded to $O(\text{bins})$ rather than $O(2^d \cdot \text{bins})$.
- Completely replaces the $O(N)$ raw `float64` feature scan across active nodes when `len(data.Binned.Bins) > 0`.

### 3. Intra-Tree Concurrency Gating on Dataset Size
- Introduced `minSamplesForIntraTreeParallel = 4096`.
- For datasets with $N < 4,096$ (e.g. daily time-series with $N \approx 1,500$), intra-tree feature split searches run sequentially on the caller's pre-allocated `TrainWorkspace`.
- Eliminates nested goroutine scheduling latency and `sync.Pool` memory thrashing when outer callers concurrently fit multiple models/quantiles.

### 4. Exported Pre-Binned Columnar Entry Points
- Exported `TrainEnsembleColumnar(data ColumnarDataset, config EnsembleConfig) *Model` and `TrainQuantileEnsembleColumnar(data ColumnarDataset, quantile float64, config EnsembleConfig) *Model`.
- Directly reuses pre-quantized `data.Binned` without re-sorting $O(N \log N)$ floats, enabling concurrent quantile workers ($P_{10}, P_{50}, P_{90}$) to safely share a single read-only `ColumnarDataset`.
- `TrainEnsembleWithWeights` and `TrainQuantileEnsembleWithWeights` now delegate to these columnar routines.
- Evaluates post-fit metrics (`MAE`, `RMSE`, `RSquared`, `PinballLoss`) directly from columnar representation via `computeMetricsColumnar`.

## Breaking Changes & Invariants
- **100% Backward Compatible:** All existing function signatures and model serialization formats are preserved.
- **Pure-Go, Zero-CGO (`CGO_ENABLED=0`):** Full build and test verification passed with CGO disabled.
- **Zero Allocations in Inner Tree Building:** Verified via benchmark alloc tests (`TestSymmetricLevelHistogram_DoesNotAllocate`).
- **Monotonicity & Parity:** Preserved across all quantile estimators.

## Testing & Verification Scorecard
- **License Headers:** Verified (`make check-license`).
- **Formatting & Vet:** Verified clean (`gofmt -s -w .`, `go vet ./...`).
- **Full Race Detector Suite:** 100% pass (`go test -v -race ./...`).
- **Adversarial Fuzz Testing:** All 13 fuzz targets passed with 5-second fuzzing time (`make fuzz FUZZTIME=5s`).
- **Downstream Verification:**
  - `agilecast`: All unit/integration tests pass.
  - `optigrid`: All unit/integration/e2e tests pass.
  - `lngcast`: All model and forecast tests pass.

## Installation
```bash
go get github.com/Noddy76/treeboost@v0.7.0
```
