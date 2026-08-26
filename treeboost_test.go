package treeboost

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestNode_IsLeaf(t *testing.T) {
	leaf := Node{NodeID: 1, SplitFeature: -1, LeafValue: 42.0}
	if !leaf.IsLeaf() {
		t.Errorf("Expected node with SplitFeature == -1 to be leaf")
	}

	splitNode := Node{NodeID: 0, SplitFeature: 0, SplitValue: 10.0, LeftChild: 1, RightChild: 2}
	if splitNode.IsLeaf() {
		t.Errorf("Expected node with SplitFeature >= 0 to not be leaf")
	}
}

func TestEvaluateTree(t *testing.T) {
	nodes := []Node{
		{NodeID: 0, SplitFeature: 0, SplitValue: 5.0, LeftChild: 1, RightChild: 2},
		{NodeID: 1, SplitFeature: -1, LeafValue: 10.0},
		{NodeID: 2, SplitFeature: -1, LeafValue: 20.0},
	}

	resLeft := EvaluateTree(nodes, FeatureVector{Values: []float64{3.0}})
	if resLeft != 10.0 {
		t.Errorf("Expected left leaf value 10.0, got %f", resLeft)
	}

	resRight := EvaluateTree(nodes, FeatureVector{Values: []float64{7.0}})
	if resRight != 20.0 {
		t.Errorf("Expected right leaf value 20.0, got %f", resRight)
	}
}

func TestComputeSampleWeights(t *testing.T) {
	Y := []float64{10.0, 10.0, 10.0, 100.0} // Mean is 32.5
	weights := ComputeSampleWeights(Y)

	if len(weights) != len(Y) {
		t.Fatalf("Expected weights length %d, got %d", len(Y), len(weights))
	}

	// For y = 100, diff = 67.5 => 5 * log10(77.5) - 4 ~= 5 * 1.889 - 4 = 5.44 => rounded to 5
	// Outlier should receive higher weight than mean samples
	if weights[3] <= weights[0] {
		t.Errorf("Expected outlier sample weight (%f) to be strictly greater than normal sample weight (%f)", weights[3], weights[0])
	}
}

func TestTrainSymmetricGBDT(t *testing.T) {
	// Synthetic dataset: Y = 2 * X0
	X := []FeatureVector{
		{Values: []float64{1.0}},
		{Values: []float64{2.0}},
		{Values: []float64{3.0}},
		{Values: []float64{4.0}},
		{Values: []float64{5.0}},
	}
	Y := []float64{2.0, 4.0, 6.0, 8.0, 10.0}
	W := []float64{1.0, 1.0, 1.0, 1.0, 1.0}

	params := CatBoostParams{
		Depth:        2,
		Iterations:   50,
		LearningRate: 0.1,
		L2LeafReg:    1.0,
	}

	trees, baseVal := TrainSymmetricGBDT(X, Y, W, params)
	if len(trees) != 50 {
		t.Fatalf("Expected 50 trees, got %d", len(trees))
	}

	model := &Model{
		BaseValue:    baseVal,
		Trees:        trees,
		LearningRate: 0.1,
	}

	pred := model.Predict([]float64{3.0})
	if math.Abs(pred-6.0) > 1.0 {
		t.Errorf("Expected prediction near 6.0 for X0=3.0, got %f", pred)
	}
}

func TestTrainLeafwiseGBDT(t *testing.T) {
	X := []FeatureVector{
		{Values: []float64{10.0}},
		{Values: []float64{20.0}},
		{Values: []float64{30.0}},
		{Values: []float64{40.0}},
	}
	Y := []float64{1.0, 2.0, 3.0, 4.0}
	W := []float64{1.0, 1.0, 1.0, 1.0}

	params := LightGBMParams{
		MaxDepth:        3,
		MaxLeaves:       8,
		MinChildSamples: 1,
		Estimators:      40,
		LearningRate:    0.1,
	}

	trees, baseVal := TrainLeafwiseGBDT(X, Y, W, params)
	if len(trees) != 40 {
		t.Fatalf("Expected 40 trees, got %d", len(trees))
	}

	model := &Model{
		BaseValue:    baseVal,
		Trees:        trees,
		LearningRate: 0.1,
	}

	pred := model.Predict([]float64{20.0})
	if math.Abs(pred-2.0) > 1.0 {
		t.Errorf("Expected prediction near 2.0 for X0=20.0, got %f", pred)
	}
}

func TestTrainExtraTrees(t *testing.T) {
	X := []FeatureVector{
		{Values: []float64{0.0, 1.0}},
		{Values: []float64{1.0, 0.0}},
		{Values: []float64{2.0, 2.0}},
	}
	Y := []float64{10.0, 20.0, 30.0}
	W := []float64{1.0, 1.0, 1.0}

	params := ExtraTreesParams{
		Estimators:     20,
		MinSamplesLeaf: 1,
		MaxFeatures:    1.0,
	}

	trees := TrainExtraTrees(X, Y, W, params)
	if len(trees) != 20 {
		t.Fatalf("Expected 20 extra trees, got %d", len(trees))
	}
}

func TestTrainEnsemble(t *testing.T) {
	X := []FeatureVector{
		{Values: []float64{1.0, 10.0}},
		{Values: []float64{2.0, 20.0}},
		{Values: []float64{3.0, 30.0}},
		{Values: []float64{4.0, 40.0}},
		{Values: []float64{5.0, 50.0}},
	}
	Y := []float64{15.0, 25.0, 35.0, 45.0, 55.0}

	config := EnsembleConfig{
		CatBoostWeight:   0.4,
		LightGBMWeight:   0.4,
		ExtraTreesWeight: 0.2,
		CatBoost: CatBoostParams{
			Depth:        3,
			Iterations:   30,
			LearningRate: 0.1,
			L2LeafReg:    1.0,
		},
		LightGBM: LightGBMParams{
			MaxDepth:        3,
			MaxLeaves:       8,
			MinChildSamples: 1,
			Estimators:      30,
			LearningRate:    0.1,
		},
		ExtraTrees: ExtraTreesParams{
			Estimators:     20,
			MinSamplesLeaf: 1,
			MaxFeatures:    1.0,
		},
	}

	model := TrainEnsemble(X, Y, config)
	if model == nil {
		t.Fatalf("Expected non-nil model from TrainEnsemble")
	}

	pred := model.Predict([]float64{3.0, 30.0})
	if math.IsNaN(pred) || math.IsInf(pred, 0) {
		t.Errorf("Prediction returned invalid float value: %f", pred)
	}
}

func TestSaveAndLoadModel(t *testing.T) {
	model := &Model{
		BaseValue:    12.34,
		LearningRate: 0.05,
		Trees: []Tree{
			{
				Nodes: []Node{
					{NodeID: 0, SplitFeature: 0, SplitValue: 2.5, LeftChild: 1, RightChild: 2},
					{NodeID: 1, SplitFeature: -1, LeafValue: 1.0},
					{NodeID: 2, SplitFeature: -1, LeafValue: 5.0},
				},
			},
		},
		MAE:      0.12,
		RMSE:     0.25,
		RSquared: 0.98,
	}

	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.json")

	if err := model.SaveModel(path); err != nil {
		t.Fatalf("SaveModel failed: %v", err)
	}

	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Fatalf("Model file was not created at %s", path)
	}

	loaded, err := LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}

	if loaded.BaseValue != model.BaseValue {
		t.Errorf("Expected BaseValue %f, got %f", model.BaseValue, loaded.BaseValue)
	}

	if len(loaded.Trees) != len(model.Trees) {
		t.Fatalf("Expected %d trees, got %d", len(model.Trees), len(loaded.Trees))
	}

	if loaded.Predict([]float64{1.0}) != model.Predict([]float64{1.0}) {
		t.Errorf("Loaded model prediction mismatch")
	}
}

func TestEdgeCases(t *testing.T) {
	// Nil model prediction
	var nilModel *Model
	if nilModel.Predict([]float64{1.0, 2.0}) != 0.0 {
		t.Errorf("Expected 0.0 prediction for nil model")
	}

	// Empty dataset for sample weights
	if weights := ComputeSampleWeights(nil); weights != nil {
		t.Errorf("Expected nil sample weights for empty input slice")
	}

	// EvaluateTree out-of-bounds node index handling
	badNodes := []Node{
		{NodeID: 0, SplitFeature: 0, SplitValue: 10.0, LeftChild: 99, RightChild: 100},
	}
	if val := EvaluateTree(badNodes, FeatureVector{Values: []float64{5.0}}); val != 0.0 {
		t.Errorf("Expected 0.0 for invalid child index traversal")
	}
}

func TestScaleTreeLeaves(t *testing.T) {
	tree := Tree{
		Nodes: []Node{
			{NodeID: 0, SplitFeature: 0, SplitValue: 5.0, LeftChild: 1, RightChild: 2},
			{NodeID: 1, SplitFeature: -1, LeafValue: 10.0},
			{NodeID: 2, SplitFeature: -1, LeafValue: 20.0},
		},
	}

	scaled := ScaleTreeLeaves(tree, 0.5)

	if scaled.Nodes[0].SplitFeature != 0 {
		t.Errorf("Expected root split feature to remain unchanged")
	}
	if scaled.Nodes[1].LeafValue != 5.0 {
		t.Errorf("Expected scaled leaf 1 value 5.0, got %f", scaled.Nodes[1].LeafValue)
	}
	if scaled.Nodes[2].LeafValue != 10.0 {
		t.Errorf("Expected scaled leaf 2 value 10.0, got %f", scaled.Nodes[2].LeafValue)
	}
}

func TestComputeWeightedError(t *testing.T) {
	Y := []float64{2.0, 4.0, 6.0}
	W := []float64{1.0, 1.0, 1.0}
	indices := []int{0, 1, 2}

	errSq, mean, sumW := computeWeightedError(Y, W, indices)
	if sumW != 3.0 {
		t.Errorf("Expected sumW == 3.0, got %f", sumW)
	}
	if math.Abs(mean-4.0) > 1e-6 {
		t.Errorf("Expected mean == 4.0, got %f", mean)
	}
	// diffs: -2, 0, 2 => errorSq = 4 + 0 + 4 = 8.0
	if math.Abs(errSq-8.0) > 1e-6 {
		t.Errorf("Expected errorSq == 8.0, got %f", errSq)
	}

	// Empty indices case
	errSqEmpty, meanEmpty, sumWEmpty := computeWeightedError(Y, W, nil)
	if errSqEmpty != 0.0 || meanEmpty != 0.0 || sumWEmpty != 0.0 {
		t.Errorf("Expected zeros for empty indices")
	}

	// Zero weights case
	WZero := []float64{0.0, 0.0, 0.0}
	_, _, sumWZero := computeWeightedError(Y, WZero, indices)
	if sumWZero != 0.0 {
		t.Errorf("Expected sumW == 0.0 for zero weights")
	}
}

func TestGetSplitCandidates(t *testing.T) {
	X := []FeatureVector{
		{Values: []float64{10.0}},
		{Values: []float64{20.0}},
		{Values: []float64{30.0}},
	}

	splits := getSplitCandidates(X, []int{0, 1, 2}, 0, nil)
	if len(splits) != 50 {
		t.Fatalf("Expected 50 split candidates, got %d", len(splits))
	}
	if splits[0] <= 10.0 || splits[49] >= 30.0 {
		t.Errorf("Split candidates out of range: min %f, max %f", splits[0], splits[49])
	}

	// Identical feature values -> no splits
	XSame := []FeatureVector{
		{Values: []float64{5.0}},
		{Values: []float64{5.0}},
	}
	if sameSplits := getSplitCandidates(XSame, []int{0, 1}, 0, nil); sameSplits != nil {
		t.Errorf("Expected nil split candidates for identical feature values")
	}

	// Empty indices
	if emptySplits := getSplitCandidates(X, nil, 0, nil); emptySplits != nil {
		t.Errorf("Expected nil split candidates for empty indices")
	}
}

func TestModelPredictOptions(t *testing.T) {
	// LearningRate == 0.0 defaults to 1.0
	modelZeroLR := &Model{
		BaseValue:    5.0,
		LearningRate: 0.0,
		Trees: []Tree{
			{
				Nodes: []Node{
					{NodeID: 0, SplitFeature: -1, LeafValue: 3.0},
				},
			},
			{
				Nodes: nil, // empty tree should be skipped
			},
		},
	}

	pred := modelZeroLR.Predict([]float64{1.0})
	if pred != 8.0 {
		t.Errorf("Expected prediction 8.0 (5.0 + 1.0 * 3.0), got %f", pred)
	}
}

func TestLoadModelErrors(t *testing.T) {
	// Non-existent file
	if _, err := LoadModel("/invalid/path/to/nonexistent_model.json"); err == nil {
		t.Errorf("Expected error for non-existent model file")
	}

	// Invalid JSON format
	tmpDir := t.TempDir()
	invalidPath := filepath.Join(tmpDir, "bad.json")
	if err := os.WriteFile(invalidPath, []byte("{invalid json"), 0644); err != nil {
		t.Fatalf("Failed to write temp file: %v", err)
	}

	if _, err := LoadModel(invalidPath); err == nil {
		t.Errorf("Expected unmarshal error for corrupt JSON file")
	}
}

func TestDefaultEnsembleConfigValues(t *testing.T) {
	cfg := DefaultEnsembleConfig()
	if cfg.CatBoostWeight <= 0 || cfg.LightGBMWeight <= 0 || cfg.ExtraTreesWeight <= 0 {
		t.Errorf("Default ensemble weights must be positive")
	}
	if cfg.CatBoost.Iterations != 500 || cfg.LightGBM.Estimators != 500 || cfg.ExtraTrees.Estimators != 700 {
		t.Errorf("Default estimators count mismatch")
	}
}

func BenchmarkGBDTTrainEnsemble(b *testing.B) {
	// Generate 500 synthetic samples with 10 features
	nSamples := 500
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

	config := DefaultEnsembleConfig()

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		model := TrainEnsemble(X, Y, config)
		if model == nil {
			b.Fatal("TrainEnsemble returned nil")
		}
	}
}

func BenchmarkGBDTPredict(b *testing.B) {
	nSamples := 500
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
	config := DefaultEnsembleConfig()
	config.CatBoost.Iterations = 50
	config.LightGBM.Estimators = 50
	config.ExtraTrees.Estimators = 50
	model := TrainEnsemble(X, Y, config)

	testFeatures := []float64{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = model.Predict(testFeatures)
	}
}

func TestComputeWeightedSum_Parity(t *testing.T) {
	n := 127
	W := make([]float64, n)
	Y := make([]float64, n)
	indices := make([]int, n)

	for i := 0; i < n; i++ {
		W[i] = float64(i+1) * 0.5
		Y[i] = float64((i*17)%31) - 15.0
		indices[i] = i
	}

	sumWScalar, sumWYScalar := computeWeightedSumScalar(W, Y, indices)
	sumWDisp, sumWYDisp := computeWeightedSum(W, Y, indices)

	if math.Abs(sumWScalar-sumWDisp) > 1e-6 {
		t.Errorf("sumW mismatch: scalar %f, dispatch %f", sumWScalar, sumWDisp)
	}
	if math.Abs(sumWYScalar-sumWYDisp) > 1e-6 {
		t.Errorf("sumWY mismatch: scalar %f, dispatch %f", sumWYScalar, sumWYDisp)
	}
}

func TestContiguousWeightedSum_Parity(t *testing.T) {
	testSizes := []int{0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 64, 127, 256, 1024}
	for _, n := range testSizes {
		W := make([]float64, n)
		Y := make([]float64, n)
		for i := 0; i < n; i++ {
			W[i] = float64(i+1)*0.25 + 0.1
			Y[i] = float64((i*23)%47) - 20.0
		}

		sumWScalar, sumWYScalar := computeWeightedSumContiguousScalar(W, Y)
		sumWDisp, sumWYDisp := computeWeightedSumContiguous(W, Y)

		if math.Abs(sumWScalar-sumWDisp) > 1e-5 {
			t.Errorf("size %d sumW mismatch: scalar %f, dispatch %f", n, sumWScalar, sumWDisp)
		}
		if math.Abs(sumWYScalar-sumWYDisp) > 1e-5 {
			t.Errorf("size %d sumWY mismatch: scalar %f, dispatch %f", n, sumWYScalar, sumWYDisp)
		}
	}
}

func TestInPlacePartitioning(t *testing.T) {
	features := []float64{10.0, 50.0, 20.0, 80.0, 30.0, 90.0, 40.0}
	sampleIdxs := []int{0, 1, 2, 3, 4, 5, 6}

	splitVal := 45.0
	mid := partitionSamplesInPlace(sampleIdxs, 0, len(sampleIdxs), features, 0, splitVal)

	for i := 0; i < mid; i++ {
		val := features[sampleIdxs[i]]
		if val >= splitVal {
			t.Errorf("Left partition index %d value %f >= splitVal %f", sampleIdxs[i], val, splitVal)
		}
	}
	for i := mid; i < len(sampleIdxs); i++ {
		val := features[sampleIdxs[i]]
		if val < splitVal {
			t.Errorf("Right partition index %d value %f < splitVal %f", sampleIdxs[i], val, splitVal)
		}
	}
}

func TestPredictBatch_Parity(t *testing.T) {
	nSamples := 64
	nFeatures := 10
	X := make([]FeatureVector, nSamples)
	Y := make([]float64, nSamples)
	var flatFeatures []float64

	for i := 0; i < nSamples; i++ {
		vals := make([]float64, nFeatures)
		for f := 0; f < nFeatures; f++ {
			vals[f] = float64((i*19 + f*7) % 50)
		}
		X[i] = FeatureVector{Values: vals}
		Y[i] = vals[0]*1.5 + vals[1]*0.8
		flatFeatures = append(flatFeatures, vals...)
	}

	config := DefaultEnsembleConfig()
	config.CatBoost.Iterations = 20
	config.LightGBM.Estimators = 20
	config.ExtraTrees.Estimators = 20
	model := TrainEnsemble(X, Y, config)

	batchPreds := model.PredictBatch(flatFeatures, nSamples)
	if len(batchPreds) != nSamples {
		t.Fatalf("Expected %d batch predictions, got %d", nSamples, len(batchPreds))
	}

	for i := 0; i < nSamples; i++ {
		singlePred := model.Predict(X[i].Values)
		if math.Abs(batchPreds[i]-singlePred) > 1e-6 {
			t.Errorf("Sample %d prediction mismatch: single %f, batch %f", i, singlePred, batchPreds[i])
		}
	}
}

func BenchmarkPredictBatch(b *testing.B) {
	nSamples := 64
	nFeatures := 10
	X := make([]FeatureVector, nSamples)
	Y := make([]float64, nSamples)
	var flatFeatures []float64

	for i := 0; i < nSamples; i++ {
		vals := make([]float64, nFeatures)
		for f := 0; f < nFeatures; f++ {
			vals[f] = float64((i*19 + f*7) % 50)
		}
		X[i] = FeatureVector{Values: vals}
		Y[i] = vals[0]*1.5 + vals[1]*0.8
		flatFeatures = append(flatFeatures, vals...)
	}

	config := DefaultEnsembleConfig()
	config.CatBoost.Iterations = 50
	config.LightGBM.Estimators = 50
	config.ExtraTrees.Estimators = 50
	model := TrainEnsemble(X, Y, config)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = model.PredictBatch(flatFeatures, nSamples)
	}
}

func BenchmarkWeightedSumContiguous(b *testing.B) {
	n := 1024
	W := make([]float64, n)
	Y := make([]float64, n)
	for i := 0; i < n; i++ {
		W[i] = float64(i+1) * 0.5
		Y[i] = float64((i*17)%31) - 15.0
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = computeWeightedSumContiguous(W, Y)
	}
}

func BenchmarkWeightedSumContiguousScalar(b *testing.B) {
	n := 1024
	W := make([]float64, n)
	Y := make([]float64, n)
	for i := 0; i < n; i++ {
		W[i] = float64(i+1) * 0.5
		Y[i] = float64((i*17)%31) - 15.0
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = computeWeightedSumContiguousScalar(W, Y)
	}
}

func TestHistogramQuantizationAndSplit(t *testing.T) {
	nSamples := 100
	nFeatures := 4
	X := make([]FeatureVector, nSamples)
	Y := make([]float64, nSamples)
	W := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		vals := []float64{float64(i), float64(100 - i), float64(i % 10), float64(i * 2)}
		X[i] = FeatureVector{Values: vals}
		if i < 50 {
			Y[i] = 10.0
		} else {
			Y[i] = 100.0
		}
		W[i] = 1.0
	}

	cd := ToColumnar(X, Y, W)
	if cd.Binned.NumSamples != nSamples || cd.Binned.NumFeatures != nFeatures {
		t.Fatalf("Expected BinnedDataset samples=%d, features=%d, got samples=%d, features=%d",
			nSamples, nFeatures, cd.Binned.NumSamples, cd.Binned.NumFeatures)
	}

	var hist FeatureHistogram
	indices := make([]int, nSamples)
	for i := 0; i < nSamples; i++ {
		indices[i] = i
	}

	BuildHistogram(cd.Binned.Bins, 0, cd.Weights, cd.Targets, indices, &hist)
	splitVal, _, ok := FindBestHistogramSplit(&hist, cd.Binned.BinBoundaries[0], 100.0, 5500.0)
	if !ok {
		t.Fatalf("Expected valid histogram split for feature 0")
	}
	if splitVal < 40.0 || splitVal > 60.0 {
		t.Errorf("Expected split threshold around 49-50, got %f", splitVal)
	}
}

// TestNonlinearFunctionApproximation tests GBDT ensemble fitting on non-linear multi-variable functions.
func TestNonlinearFunctionApproximation(t *testing.T) {
	nSamples := 300
	X := make([]FeatureVector, nSamples)
	Y := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		x0 := (float64(i) / float64(nSamples)) * 4.0 * math.Pi // 0 to 4pi
		x1 := float64(i%20) / 5.0                             // 0 to 4
		X[i] = FeatureVector{Values: []float64{x0, x1}}
		Y[i] = math.Sin(x0) + x1*x1
	}

	config := DefaultEnsembleConfig()
	config.CatBoost.Iterations = 150
	config.LightGBM.Estimators = 150
	config.ExtraTrees.Estimators = 150

	model := TrainEnsemble(X, Y, config)
	if model == nil {
		t.Fatalf("TrainEnsemble returned nil model")
	}

	if model.RSquared < 0.85 {
		t.Errorf("Expected R^2 >= 0.85 for non-linear fit, got %f (MAE: %f, RMSE: %f)",
			model.RSquared, model.MAE, model.RMSE)
	}

	// Test prediction on unseen test sample
	testPoint := []float64{math.Pi / 2.0, 2.0} // sin(pi/2) + 2^2 = 1 + 4 = 5
	pred := model.Predict(testPoint)
	if math.Abs(pred-5.0) > 1.5 {
		t.Errorf("Expected prediction near 5.0 for sin(pi/2)+4, got %f", pred)
	}
}

// TestConcurrentPredictions verifies that trained Model is thread-safe for concurrent read access.
func TestConcurrentPredictions(t *testing.T) {
	nSamples := 100
	nFeatures := 5
	X := make([]FeatureVector, nSamples)
	Y := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		vals := make([]float64, nFeatures)
		for f := 0; f < nFeatures; f++ {
			vals[f] = float64((i*11 + f*7) % 50)
		}
		X[i] = FeatureVector{Values: vals}
		Y[i] = vals[0] + vals[1]*2.0
	}

	config := DefaultEnsembleConfig()
	config.CatBoost.Iterations = 30
	config.LightGBM.Estimators = 30
	config.ExtraTrees.Estimators = 30
	model := TrainEnsemble(X, Y, config)

	const numGoroutines = 50
	const iterationsPerGoroutine = 100
	done := make(chan bool, numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		go func(gID int) {
			for iter := 0; iter < iterationsPerGoroutine; iter++ {
				idx := (gID + iter) % nSamples
				pred := model.Predict(X[idx].Values)
				if math.IsNaN(pred) || math.IsInf(pred, 0) {
					t.Errorf("Goroutine %d produced invalid prediction: %f", gID, pred)
				}
			}
			done <- true
		}(g)
	}

	for g := 0; g < numGoroutines; g++ {
		<-done
	}
}

// TestSpecialFloatValuesAndBoundaries verifies graceful behavior with zero variance targets,
// uniform feature values, and extreme float inputs.
func TestSpecialFloatValuesAndBoundaries(t *testing.T) {
	// Zero variance targets (all targets = 42.0)
	XUniform := []FeatureVector{
		{Values: []float64{1.0, 2.0}},
		{Values: []float64{3.0, 4.0}},
		{Values: []float64{5.0, 6.0}},
	}
	YUniform := []float64{42.0, 42.0, 42.0}

	config := DefaultEnsembleConfig()
	config.CatBoost.Iterations = 10
	config.LightGBM.Estimators = 10
	config.ExtraTrees.Estimators = 10
	modelUniform := TrainEnsemble(XUniform, YUniform, config)

	predUniform := modelUniform.Predict([]float64{2.0, 3.0})
	if math.Abs(predUniform-42.0) > 1e-4 {
		t.Errorf("Expected prediction near 42.0 for uniform target, got %f", predUniform)
	}

	// Mismatched feature dimensions in Predict
	predMismatched := modelUniform.Predict([]float64{1.0}) // pass 1 feature instead of 2
	if math.IsNaN(predMismatched) || math.IsInf(predMismatched, 0) {
		t.Errorf("Predict returned invalid value for mismatched feature count: %f", predMismatched)
	}

	// Empty features in PredictBatch
	batchEmpty := modelUniform.PredictBatch(nil, 0)
	if len(batchEmpty) != 0 {
		t.Errorf("Expected empty batch slice for 0 samples")
	}
}

// TestQuantizationEdgeCases tests binning when sample count is less than maxBins or features are identical.
func TestQuantizationEdgeCases(t *testing.T) {
	// 3 samples binned into max 256 bins
	XSmall := []FeatureVector{
		{Values: []float64{10.0, 5.0}},
		{Values: []float64{20.0, 5.0}}, // feature 1 is identical
		{Values: []float64{30.0, 5.0}},
	}
	YSmall := []float64{1.0, 2.0, 3.0}
	WSmall := []float64{1.0, 1.0, 1.0}

	cd := ToColumnar(XSmall, YSmall, WSmall)
	if cd.Binned.NumSamples != 3 || cd.Binned.NumFeatures != 2 {
		t.Fatalf("Expected NumSamples=3, NumFeatures=2, got %d, %d", cd.Binned.NumSamples, cd.Binned.NumFeatures)
	}
	if cd.Binned.BinCounts[1] != 1 {
		t.Errorf("Expected 1 unique bin boundary for constant feature 1, got %d", cd.Binned.BinCounts[1])
	}
}

// TestFullEnsembleSerializationRoundtrip tests JSON marshalling and unmarshalling of complex multi-algorithm models.
func TestFullEnsembleSerializationRoundtrip(t *testing.T) {
	X := []FeatureVector{
		{Values: []float64{1.0, 5.0}},
		{Values: []float64{2.0, 10.0}},
		{Values: []float64{3.0, 15.0}},
		{Values: []float64{4.0, 20.0}},
	}
	Y := []float64{10.0, 20.0, 30.0, 40.0}

	config := DefaultEnsembleConfig()
	config.CatBoost.Iterations = 20
	config.LightGBM.Estimators = 20
	config.ExtraTrees.Estimators = 20

	model := TrainEnsemble(X, Y, config)
	model.HorizonBins = []string{"T1", "T2"}
	model.P10Offsets = []float64{-1.5, -2.5}
	model.P90Offsets = []float64{1.5, 2.5}

	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "full_ensemble.json")

	if err := model.SaveModel(path); err != nil {
		t.Fatalf("SaveModel failed: %v", err)
	}

	loaded, err := LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}

	if len(loaded.HorizonBins) != 2 || loaded.HorizonBins[0] != "T1" {
		t.Errorf("HorizonBins mismatch in loaded model")
	}
	if len(loaded.P10Offsets) != 2 || loaded.P10Offsets[0] != -1.5 {
		t.Errorf("P10Offsets mismatch in loaded model")
	}

	predOriginal := model.Predict([]float64{2.5, 12.5})
	predLoaded := loaded.Predict([]float64{2.5, 12.5})
	if math.Abs(predOriginal-predLoaded) > 1e-6 {
		t.Errorf("Loaded model prediction (%f) does not match original (%f)", predLoaded, predOriginal)
	}
}
