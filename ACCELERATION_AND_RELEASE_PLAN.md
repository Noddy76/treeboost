<!--
Copyright 2026 James Grant. All rights reserved.
-->

# Plan & Prompt Specification: Upstream Engine Acceleration, Histogram Quantization & Release (treeboost)

> **Repository:** `/home/noddy/src/treeboost` (`github.com/noddy76/treeboost`)
> **Target Release:** `v0.6.0`
> **Target Execution Environment:** Dual Intel Xeon E5-2699 v3 (32 vCPUs, NUMA Node 0/1)
> **Goal:** Introduce discrete $O(B)$ histogram binning ($B \le 256$), intra-tree feature subsampling (`ColsampleByTree` / `ColsampleByNode`), early stopping on validation quantile/pinball loss, and fine-grained intra-tree parallelism with zero heap allocation during boosting. Verify exhaustively via unit tests, downstream consumer tests, and adversarial fuzz testing, followed by a formal tag, GitHub release, and CI verification.

---

## 1. Technical Objectives & Engine Invariants

1. **Pre-Binned Discrete Histogram Split Search ($O(B)$ where $B \le 256$):**
   - Discretize continuous feature vectors into $B = 256$ quantile bins (representable as `uint8`) during dataset preparation / design matrix creation.
   - For each feature, constructing gradient/weight histograms takes $O(N)$ operations. Finding the optimal split requires scanning only $B = 256$ bins ($O(B)$) instead of sorting or scanning hundreds of continuous `float64` candidates in `getSplitCandidatesColumnar`.
   - Maintain strict `uint8` cache locality (1 byte vs 8 bytes per sample), eliminating NUMA memory bus bottlenecks across 32 vCPUs.
2. **Feature Subsampling (`ColsampleByTree` & `ColsampleByNode`):**
   - Introduce `ColsampleByTree float64` and `ColsampleByNode float64` in `EnsembleConfig`.
   - Subsample feature subsets per tree or per node without allocating new slices on the heap (using workspace index pools).
   - Halves the number of feature histogram evaluations per split.
3. **Early Stopping on Validation Pinball / Quantile Loss:**
   - Add early stopping support to `TrainQuantileEnsemble` and `TrainEnsembleWithWeights` based on a validation set and a configurable `EarlyStoppingPatience int` (e.g. 15 iterations).
   - Stop boosting iterations early when validation pinball loss $\min_{i \le t} L_{\text{pinball}}(i)$ has not improved for the patience window.
4. **Intra-Tree Parallel Split Search:**
   - Parallelize the feature histogram aggregation loop across worker goroutines when CPU worker threads exceed concurrent model count.
   - Workers write into thread-local histogram buffers and combine via lock-free or deterministic reduction, adhering to the zero-allocation workspace invariant.
5. **Hard Invariants & Zero-Regression Backward Compatibility:**
   - **Pure-Go, Zero-CGO (`CGO_ENABLED=0`):** Build and test cleanly with `CGO_ENABLED=0`. Hardware acceleration uses native Go assembly (`weighted_sum_amd64.s`) with runtime detection and scalar fallbacks.
   - **Zero-Allocation Inner Boosting Loops:** Reuse slice buffers in `TrainWorkspace` acquired via `acquireWorkspace` and released via `releaseWorkspace`.
   - **Fail-Fast Parameter Validation:** Reject non-finite, out-of-bounds, or mismatched inputs immediately with descriptive errors or panics.
   - **Monotonicity & Boundary Preservation:** Enforce $Q_{10} \le Q_{50} \le Q_{90}$ and exact parity with quantized histogram boundaries.
   - **Zero Regression on Existing Default Behavior (100% Backward Compatibility):**
     - When `ColsampleByTree` / `ColsampleByNode` are `0` or unset, default to `1.0` (all features evaluated, identical to previous behavior).
     - When `EarlyStoppingPatience` is `0` or unset (or validation set is nil), default to running the full iteration budget unconditionally.
     - Existing continuous floating-point evaluation pathways or default histogram quantizers must yield identical split boundaries and identical model outputs for unbinned continuous inputs.
     - All existing downstream tests in sibling clients (`agilecast`, `optigrid`, `lngcast`) must pass without changing their results.

---

## 2. Comprehensive Verification Plan

In accordance with `/home/noddy/src/treeboost/AGENTS.md`:

### 2.1 License & Static Analysis
- `make check-license` (ensure Apache 2.0 and Copyright 2026 James Grant on all files)
- `gofmt -s -w .`
- `go vet ./...`

### 2.2 Race-Free Testing
- `CGO_ENABLED=0 go test -v -race ./...`

### 2.3 Adversarial Fuzz Testing
- Run existing 13 fuzz targets plus new fuzz targets for histogram binning, colsample, and early stopping:
  ```bash
  make fuzz FUZZTIME=5s
  ```
- **New Adversarial Fuzz Targets to Add:**
  - `FuzzQuantizeDataset_HistogramBinning`: Fuzz arbitrary float matrices (NaNs, +Inf, -Inf, subnormals, identical constants, extreme magnitudes) through the new binning and split search.
  - `FuzzEnsembleEarlyStopping`: Fuzz training with random patience, malformed validation matrices, zero variance targets, and extreme learning rates.
  - `FuzzColsampleFeatureSelection`: Fuzz feature subsampling ratios (`colsample <= 0`, `> 1`, `NaN`, 0 features selected, all features selected) to ensure strict parameter validation and zero panic on boundary edge cases.

### 2.4 Downstream Client Verification
Verify 100% backward compatibility and zero regressions against downstream consumers:
- `(cd /home/noddy/src/agilecast && go test ./...)`
- `(cd /home/noddy/src/optigrid && go test ./...)`
- `(cd /home/noddy/src/lngcast && go test ./...)`

---

## 3. Release Protocol & GitHub Release Notes Specification

Upon full verification:
1. Commit with a conventional commit message:
   ```bash
   git add <files>
   git commit -m "feat: v0.6.0 histogram binning, feature subsampling, early stopping, and intra-tree acceleration"
   ```
2. Create annotated git tag:
   ```bash
   git tag -a v0.6.0 -m "Release v0.6.0: Histogram binning, colsample, early stopping, and multi-core split search"
   git push origin main
   git push origin v0.6.0
   ```
3. Author formal GitHub Release Notes file (`RELEASE_NOTES_v0.6.0.md`) covering:
   - **Overview & Motivation:** Dual Xeon multi-core scaling, eliminating continuous split scans, reducing memory bandwidth by 8x.
   - **Key Capabilities & Algorithmic Implementations:**
     - Quantile-based `uint8` feature binning and $O(256)$ histogram split finding.
     - `ColsampleByTree` and `ColsampleByNode` feature subsampling.
     - Validation-based early stopping on pinball loss ($L_\tau$).
     - Fine-grained intra-tree multi-threading with thread-local histogram pooling.
   - **Numerical Safety & Invariants:** Zero-CGO, zero heap allocation in inner loops, leaf value bounding, strict input validation.
   - **Testing & Verification Scorecard:** Full race-detector suite pass, fuzz testing iterations (including new fuzzers), downstream client regression check (`agilecast`, `optigrid`, `lngcast`).
   - **Installation Instructions:** `go get github.com/noddy76/treeboost@v0.6.0`.
4. Publish release via GitHub CLI:
   ```bash
   gh release create v0.6.0 --title "v0.6.0: Histogram Binning, Feature Subsampling & Early Stopping Acceleration" --notes-file RELEASE_NOTES_v0.6.0.md
   gh release view v0.6.0
   ```
5. Monitor upstream CI workflow:
   ```bash
   gh run list --limit 3
   gh run watch <RUN_ID>
   ```

---

## 4. Ready-to-Use Agent Prompt (`treeboost`)

```markdown
Implement the upstream engine acceleration, histogram quantization, feature subsampling, early stopping, adversarial fuzz testing, and formal v0.6.0 release in /home/noddy/src/treeboost as specified in ACCELERATION_AND_RELEASE_PLAN.md.

Context & Rules:
- Repository: /home/noddy/src/treeboost (GitHub: github.com/noddy76/treeboost)
- Target Release: v0.6.0
- Target Host: Dual Intel Xeon E5-2699 v3 (32 container vCPUs, NUMA Node 0/1)
- Mandatory Guidelines: Follow /home/noddy/src/treeboost/AGENTS.md strictly. Never stop at git tagging; a formal GitHub release with structured markdown notes is mandatory.

Core Implementation Tasks:
1. Implement uint8 histogram binning for continuous features (O(256) split evaluation replacing O(N*K) continuous candidate scanning).
2. Implement ColsampleByTree and ColsampleByNode in EnsembleConfig with zero-allocation workspace pooling.
3. Add validation early stopping on pinball loss with configurable patience (EarlyStoppingPatience).
4. Parallelize the intra-tree feature histogram accumulation loop across worker goroutines using thread-local workspace pools.
5. Implement new adversarial fuzz tests in treeboost:
   - FuzzQuantizeDataset_HistogramBinning
   - FuzzEnsembleEarlyStopping
   - FuzzColsampleFeatureSelection
6. Run all pre-release verification checks:
   - make check-license
   - gofmt -s -w . && go vet ./...
   - CGO_ENABLED=0 go test -v -race ./...
   - make fuzz FUZZTIME=5s (all fuzzers must pass with zero panics/leaks)
   - Downstream client tests: (cd /home/noddy/src/agilecast && go test ./...), (cd /home/noddy/src/optigrid && go test ./...), (cd /home/noddy/src/lngcast && go test ./...)
7. Tag and Release:
   - git commit and tag v0.6.0
   - git push origin main && git push origin v0.6.0
   - Create RELEASE_NOTES_v0.6.0.md and publish via gh release create v0.6.0
   - Verify upstream CI via gh run watch
```
