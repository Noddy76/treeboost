# Treeboost Agent Guidelines & Operational Policies

This document establishes mandatory operational instructions, code quality standards, numerical safety invariants, and the **mandatory release protocol** for all autonomous AI agents working in this repository.

---

## 1. Mandatory Release Protocol (Cutting a New Release)

> [!CAUTION]
> **Never Stop at Git Tagging:**
> A release is **NOT** complete when a `git tag` is created or pushed. Every release must include a formal GitHub Release with comprehensive release notes. Pushing a tag without creating the corresponding GitHub release is a defect.

Whenever cutting a new release (`vX.Y.Z`):

### 1.1 Pre-Release Verification Checklist
Before creating a tag, ensure all the following pass with zero errors or warnings:
1. **License Headers:**
   ```bash
   make check-license
   ```
   Every `.go` file must contain the Apache License 2.0 header and Copyright notice (`Copyright 2026 James Grant`).
2. **Formatting & Static Analysis:**
   ```bash
   gofmt -s -w .
   go vet ./...
   ```
3. **Full Race-Detector Test Suite:**
   ```bash
   go test -v -race ./...
   ```
4. **Adversarial Fuzz Testing:**
   ```bash
   make fuzz FUZZTIME=5s
   ```
   All 13 fuzz targets must execute cleanly without crashes, memory leaks, or panics.
5. **Downstream Client Non-Regression Verification:**
   Run tests in local sibling repositories that consume `treeboost`:
   ```bash
   (cd /home/noddy/src/agilecast && go test ./...)
   (cd /home/noddy/src/optigrid && go test ./...)
   ```
   Ensure 100% backward compatibility and zero regressions for existing production systems.

### 1.2 Git Commit & Tagging
1. Commit all changes to `main` with a descriptive conventional commit message:
   ```bash
   git add <files>
   git commit -m "feat: vX.Y.Z <short summary>"
   ```
2. Create an annotated git tag:
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z: <short summary>"
   ```
3. Push the commit and tag to `origin main`:
   ```bash
   git push origin main
   git push origin vX.Y.Z
   ```

### 1.3 Mandatory GitHub Release Creation (`gh release create`)
Immediately after pushing the tag, author structured markdown release notes covering:
- **Overview & Motivation**
- **Key Capabilities & Algorithmic Implementations** (GBDT, QRF, SIMD, Workspace pooling)
- **Breaking Changes or Invariants** (or confirmation of 100% backward compatibility)
- **Testing & Verification Scorecard** (unit tests, coverage, monotonicity, fuzz iterations, client tests)
- **Installation instructions** (`go get github.com/Noddy76/treeboost@vX.Y.Z`)

Publish the release using the GitHub CLI:
```bash
gh release create vX.Y.Z --title "vX.Y.Z: <Title>" --notes-file <path-to-notes-file>
```
Verify the published release:
```bash
gh release view vX.Y.Z
```

### 1.4 Upstream CI Monitoring
Verify that the triggered GitHub Actions workflow on `origin main` completes with conclusion `success`:
```bash
gh run list --limit 3
gh run watch <RUN_ID>
```

---

## 2. Code Quality, SIMD, and Numerical Safety Invariants

- **Pure-Go, Zero-CGO (`CGO_ENABLED=0`):**
  `treeboost` must build and test cleanly with `CGO_ENABLED=0`. Hardware acceleration uses native Go assembly (`weighted_sum_amd64.s`) with automatic runtime CPU feature detection (`simd_amd64.go`) and scalar fallbacks (`simd_other.go`).
- **Zero-Allocation Inner Boosting Loops:**
  Split finding and leaf refitting within GBDT boosting iterations must never allocate on the heap. Always reuse slice buffers in `TrainWorkspace` (`SampleIdxs`, `PermBuf`, `SplitBuf`, `Ranges`, `Hist`, `QuantileScratch`) acquired via `acquireWorkspace` and released via `releaseWorkspace`.
- **Fail-Fast Parameter Validation & Zero Tolerance for Fabricated Data:**
  Never inject fake, synthetic, or made-up default numbers when callers provide invalid, non-finite, or out-of-bounds parameters. Non-sensical inputs (e.g., negative regularization, out-of-bounds quantiles $\tau \notin (0, 1)$, NaN/Inf values, slice length mismatches) must fail fast with immediate, descriptive panics or errors.
- **Leaf Value Bounding:**
  All tree builders (Symmetric GBDT, Leaf-wise GBDT, ExtraTrees) must clamp leaf values to `[-MaxLeafValue, +MaxLeafValue]` via `clampLeafValue` to prevent gradient explosion and divergence on noisy data.
- **Monotonicity & Coverage Guarantees:**
  Quantile models and intervals must preserve $Q_{10} \le Q_{50} \le Q_{90}$. In-place partition synchronization must guarantee strict parity with quantized histogram boundaries.
