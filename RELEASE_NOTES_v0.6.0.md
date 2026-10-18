# Release Notes v0.6.0: Histogram Binning, Feature Subsampling & Early Stopping Acceleration

`treeboost` v0.6.0 introduces major algorithmic and high-throughput training engine enhancements designed specifically for multi-core NUMA host architectures (such as dual Intel Xeon E5-2699 v3 hosts with 32 vCPUs). This release reduces memory bandwidth consumption up to 8x and accelerates split evaluation via discrete histogram binning, intra-tree feature subsampling, validation-based early stopping, and fine-grained intra-tree parallelism with strict zero heap allocation in inner boosting loops.

---

## 1. Key Capabilities & Algorithmic Implementations

### 1.1 Discrete $O(B)$ Histogram Binning ($B \le 256$, `uint8`)
- Continuous floating-point features are discretized into $B = 256$ quantile bins representable as `uint8`.
- In-tree split evaluation scans compact 256-bin gradient and weight histograms ($O(B)$) rather than continuous floating-point sample arrays ($O(N \cdot K)$).
- Reduces memory footprint from 8 bytes (`float64`) to 1 byte (`uint8`) per sample-feature, drastically improving cache locality across dual-socket NUMA nodes.

### 1.2 Feature Subsampling (`ColsampleByTree` & `ColsampleByNode`)
- Added `ColsampleByTree float64` and `ColsampleByNode float64` to `EnsembleConfig`, `CatBoostParams`, and `LightGBMParams`.
- Features are subsampled per tree or per node split without heap allocations using reusable workspace index pools and Fisher-Yates shuffling.
- Defaults to `1.0` (all features evaluated) when unset or `<= 0.0` for 100% backward compatibility.
- Enforces fail-fast parameter validation (panics on non-finite, negative, or `> 1.0` values).

### 1.3 Validation-Based Early Stopping on Pinball Loss
- Added early stopping support controlled by `EarlyStoppingPatience int` with validation dataset specifications (`ValidationData`, `ValidationX`, `ValidationY`, `ValidationW`).
- Evaluates validation pinball loss ($L_\tau$) for quantile models or MSE for mean models after each boosting iteration.
- Automatically truncates tree ensembles to the optimal iteration $\arg\min_{i \le t} L_{\text{val}}(i)$ when validation performance fails to improve over the patience window, mitigating overfitting and saving wall-clock execution time.
- Defaults to `0` (early stopping inactive) when unset.

### 1.4 Fine-Grained Intra-Tree Parallelism
- Evaluates candidate feature splits across worker goroutines via thread-local workspaces acquired from `trainWorkspacePool`.
- Employs deterministic reduction to select the global optimal split with zero heap allocation during boosting.

---

## 2. Numerical Safety & Invariants

- **Pure-Go, Zero-CGO (`CGO_ENABLED=0`):** Compiles and tests cleanly with `CGO_ENABLED=0` across all architectures, with AVX2 SIMD acceleration in Go assembly (`weighted_sum_amd64.s`) and automatic scalar fallbacks.
- **Zero-Allocation Inner Loops:** Reusable slice buffers in `TrainWorkspace` (`SampleIdxs`, `PermBuf`, `SplitBuf`, `Ranges`, `Hist`, `QuantileScratch`, `TreeFeatureIdxs`, `NodeFeatureIdxs`) eliminate GC pressure.
- **Fail-Fast Parameter Validation:** Rejects non-finite, negative, or invalid configurations immediately.
- **Strict Monotonicity & Parity:** Preserves quantile interval monotonicity ($Q_{10} \le Q_{50} \le Q_{90}$) and split parity between sequential and parallel evaluation modes.
- **100% Backward Compatibility:** Unset colsample and early stopping parameters preserve identical default behavior and split outputs across existing production models.

---

## 3. Testing & Verification Scorecard

- **License Headers:** Verified via `make check-license` (Apache 2.0 and Copyright 2026 James Grant on all Go sources).
- **Static Analysis & Formatting:** Verified via `gofmt -s -w .` and `go vet ./...` with zero issues.
- **CGO_ENABLED=0 Compatibility:** 100% passed on pure Go toolchain.
- **Race Detector Test Suite:** `go test -v -race ./...` passed with zero data races.
- **Adversarial Fuzz Testing Suite:** All 16 fuzz targets passed cleanly with zero crashes, panics, or memory leaks under `make fuzz FUZZTIME=5s`:
  - `FuzzFindBestHistogramSplit`
  - `FuzzHistogramPartitionParity`
  - `FuzzQuantizeDataset_SpecialFloats`
  - `FuzzEvaluateTree`
  - `FuzzModelDeserializationAndEvaluation`
  - `FuzzTrainEnsembleWithWeights`
  - `FuzzChronologicalCVWithOptions`
  - `FuzzComputeSampleWeights`
  - `FuzzHorizonBinAndIntervalPrediction`
  - `FuzzSIMDWeightedSumParity`
  - `FuzzWeightedQuantile`
  - `FuzzPinballLoss`
  - `FuzzTrainQuantileEnsemble`
  - `FuzzQuantizeDataset_HistogramBinning` *(new)*
  - `FuzzEnsembleEarlyStopping` *(new)*
  - `FuzzColsampleFeatureSelection` *(new)*
- **Downstream Sibling Client Non-Regression Verification:**
  - `github.com/noddy76/agilecast`: 100% test pass.
  - `git.queeg.org/queeg/optigrid`: 100% test pass.
  - `github.com/noddy76/lngcast`: 100% test pass.

---

## 4. Installation

```bash
go get github.com/noddy76/treeboost@v0.6.0
```
