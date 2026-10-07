// Copyright 2026 James Grant
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package treeboost provides a general-purpose, high-performance Gradient Boosted Decision Trees (GBDT)
// and multi-algorithm ensemble regression library in pure Go with AVX2 SIMD acceleration.
//
// Theoretical Background & Inspiration:
// The GBDT algorithms and ensemble architectures in this package are inspired by and follow the mathematical formulations detailed in:
//   - Gradient Boosting: https://en.wikipedia.org/wiki/Gradient_boosting
//   - Symmetric Decision Trees (CatBoost): https://en.wikipedia.org/wiki/CatBoost
//   - Leaf-wise Decision Trees (LightGBM): https://en.wikipedia.org/wiki/LightGBM
//   - Extremely Randomized Trees (ExtraTrees): https://en.wikipedia.org/wiki/Random_forest#ExtraTrees
package treeboost

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"
)

// Regressor defines the common interface for models capable of generating continuous scalar predictions.
type Regressor interface {
	Predict(features []float64) float64
}

// Node represents a single decision node or leaf node within a decision tree.
type Node struct {
	NodeID       int     `json:"node_id"`
	SplitFeature int     `json:"split_feature"` // -1 if leaf node; feature index 0..N-1 otherwise
	SplitValue   float64 `json:"split_value"`
	LeafValue    float64 `json:"leaf_value"`
	LeftChild    int     `json:"left_child"`
	RightChild   int     `json:"right_child"`
}

// IsLeaf returns true if the node is a leaf node.
func (n Node) IsLeaf() bool {
	return n.SplitFeature == -1
}

// Tree represents a single decision tree composed of ordered nodes.
type Tree struct {
	Nodes []Node `json:"nodes"`
}

// Evaluate evaluates a single decision tree on a feature vector.
func (t Tree) Evaluate(features []float64) float64 {
	return EvaluateTree(t.Nodes, FeatureVector{Values: features})
}

// Model represents a trained ensemble model comprising decision trees, a base initial prediction value,
// learning rate, and evaluation / diagnostic metadata.
type Model struct {
	ID              int       `json:"-"`
	TrainedAt       time.Time `json:"trained_at"`
	TrainingStart   time.Time `json:"training_start"`
	TrainingEnd     time.Time `json:"training_end"`
	BaseValue       float64   `json:"base_value"`
	Trees           []Tree    `json:"trees"`
	LearningRate    float64   `json:"learning_rate"`
	MAE             float64   `json:"mae"`
	RMSE            float64   `json:"rmse"`
	RSquared        float64   `json:"r_squared"`
	HorizonBins     []string  `json:"horizon_bins,omitempty"`
	P10Offsets      []float64 `json:"p10_offsets,omitempty"`
	P90Offsets      []float64 `json:"p90_offsets,omitempty"`
	DiagnosticsJSON string    `json:"diagnostics_json,omitempty"`
}

// Predict evaluates the GBDT model on a slice of feature values and returns the predicted scalar result.
// It initializes with the BaseValue and adds the weighted leaf prediction from each decision tree in the ensemble.
func (m *Model) Predict(features []float64) float64 {
	if m == nil {
		return 0.0
	}
	val := m.BaseValue
	lr := m.LearningRate
	if lr == 0.0 {
		lr = 1.0
	}

	fv := FeatureVector{Values: features}
	for _, tree := range m.Trees {
		if len(tree.Nodes) == 0 {
			continue
		}
		val += lr * EvaluateTree(tree.Nodes, fv)
	}

	return val
}

// ComputeMetrics evaluates MAE, RMSE, and R^2 of the Model on feature vectors X and targets Y,
// updating the model's MAE, RMSE, and RSquared fields.
func (m *Model) ComputeMetrics(X []FeatureVector, Y []float64) {
	if m == nil || len(X) == 0 || len(Y) == 0 || len(X) != len(Y) {
		return
	}
	n := float64(len(Y))
	var sumY float64
	for _, y := range Y {
		sumY += y
	}
	meanY := sumY / n

	nSamples := len(X)
	nFeatures := len(X[0].Values)
	flat := make([]float64, nSamples*nFeatures)
	for i := 0; i < nSamples; i++ {
		offset := i * nFeatures
		copy(flat[offset:offset+nFeatures], X[i].Values)
	}
	preds := m.PredictBatch(flat, nSamples)

	var maeSum, sse, sst float64
	for i := 0; i < len(Y); i++ {
		pred := preds[i]
		diff := Y[i] - pred
		maeSum += math.Abs(diff)
		sse += diff * diff
		yDiff := Y[i] - meanY
		sst += yDiff * yDiff
	}

	m.MAE = maeSum / n
	m.RMSE = math.Sqrt(sse / n)
	if sst > 1e-9 {
		m.RSquared = 1.0 - (sse / sst)
	} else {
		m.RSquared = 1.0
	}
}

// PredictBatch evaluates the GBDT model on a batch of nSamples feature vectors.
// features is a flattened slice of length (nSamples * nFeatures) in row-major layout
// (where sample i feature f is at index i*nFeatures + f).
// Returns a slice of predicted values of length nSamples.
func (m *Model) PredictBatch(features []float64, nSamples int) []float64 {
	if nSamples <= 0 {
		return nil
	}
	if m == nil || len(features) == 0 {
		return make([]float64, nSamples)
	}
	nFeatures := len(features) / nSamples
	if nFeatures == 0 {
		return make([]float64, nSamples)
	}

	lr := m.LearningRate
	if lr == 0.0 {
		lr = 1.0
	}

	results := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		results[i] = m.BaseValue
	}

	for _, tree := range m.Trees {
		nodes := tree.Nodes
		if len(nodes) == 0 {
			continue
		}

		i := 0
		for ; i+3 < nSamples; i += 4 {
			results[i] += lr * evaluateTreeFlat(nodes, features, nFeatures, i)
			results[i+1] += lr * evaluateTreeFlat(nodes, features, nFeatures, i+1)
			results[i+2] += lr * evaluateTreeFlat(nodes, features, nFeatures, i+2)
			results[i+3] += lr * evaluateTreeFlat(nodes, features, nFeatures, i+3)
		}
		for ; i < nSamples; i++ {
			results[i] += lr * evaluateTreeFlat(nodes, features, nFeatures, i)
		}
	}

	return results
}

// Standard 6 horizon bin names
var DefaultHorizonBins = []string{"0-6h", "6-12h", "12-24h", "24-36h", "36-48h", ">48h"}

// Extended 8 horizon bin names for medium-term / multi-week forecasts
var ExtendedHorizonBins = []string{"0-6h", "6-12h", "12-24h", "24-36h", "36-48h", "48-96h", "96-168h", ">168h"}

// HorizonBinForLeadTime returns the 0-indexed bin (0 to 5) for a given lead time in hours.
// Boundaries: [0, 6), [6, 12), [12, 24), [24, 36), [36, 48), [48, inf).
func HorizonBinForLeadTime(leadHours float64) int {
	switch {
	case leadHours < 6:
		return 0
	case leadHours < 12:
		return 1
	case leadHours < 24:
		return 2
	case leadHours < 36:
		return 3
	case leadHours < 48:
		return 4
	default:
		return 5
	}
}

func parseBinUpperBound(bin string) (float64, error) {
	bin = strings.TrimSpace(bin)
	bin = strings.TrimSuffix(bin, "h")
	if strings.HasPrefix(bin, ">") {
		return math.Inf(1), nil
	}
	if idx := strings.Index(bin, "-"); idx >= 0 {
		return strconv.ParseFloat(bin[idx+1:], 64)
	}
	return strconv.ParseFloat(bin, 64)
}

// HorizonBinForLeadTimeWithBins returns the 0-indexed bin for a given lead time in hours
// according to the supplied slice of horizon bin names. If bins is nil or empty, DefaultHorizonBins is used.
func HorizonBinForLeadTimeWithBins(leadHours float64, bins []string) int {
	if math.IsNaN(leadHours) || leadHours <= 0 {
		return 0
	}
	if len(bins) == 0 {
		return HorizonBinForLeadTime(leadHours)
	}
	if len(bins) == 6 && bins[5] == ">48h" {
		return HorizonBinForLeadTime(leadHours)
	}
	if len(bins) == 8 && bins[7] == ">168h" {
		switch {
		case leadHours < 6:
			return 0
		case leadHours < 12:
			return 1
		case leadHours < 24:
			return 2
		case leadHours < 36:
			return 3
		case leadHours < 48:
			return 4
		case leadHours < 96:
			return 5
		case leadHours < 168:
			return 6
		default:
			return 7
		}
	}
	for i := 0; i < len(bins)-1; i++ {
		threshold, err := parseBinUpperBound(bins[i])
		if err == nil && leadHours < threshold {
			return i
		}
	}
	return len(bins) - 1
}

// PredictInterval evaluates the model and applies empirical horizon residual offsets,
// returning (P10, P50, P90). Unconditionally guarantees P10 <= P50 <= P90 under all conditions
// (including negative targets) by clamping raw offsets.
func (m *Model) PredictInterval(features []float64, leadHours float64) (p10, p50, p90 float64) {
	if m == nil {
		return 0.0, 0.0, 0.0
	}
	p50 = m.Predict(features)
	if math.IsNaN(p50) || math.IsInf(p50, 0) {
		p50 = 0.0
	}
	bin := HorizonBinForLeadTimeWithBins(leadHours, m.HorizonBins)

	var off10, off90 float64
	if len(m.P10Offsets) > 0 {
		idx := bin
		if idx >= len(m.P10Offsets) {
			idx = len(m.P10Offsets) - 1
		}
		off10 = m.P10Offsets[idx]
	}
	if len(m.P90Offsets) > 0 {
		idx := bin
		if idx >= len(m.P90Offsets) {
			idx = len(m.P90Offsets) - 1
		}
		off90 = m.P90Offsets[idx]
	}

	if math.IsNaN(off10) || math.IsInf(off10, 0) {
		off10 = 0.0
	}
	if math.IsNaN(off90) || math.IsInf(off90, 0) {
		off90 = 0.0
	}

	p10 = math.Min(p50+off10, p50)
	p90 = math.Max(p50+off90, p50)
	return p10, p50, p90
}

// PredictBatchInterval evaluates batch feature vectors with corresponding lead times.
func (m *Model) PredictBatchInterval(flatFeatures []float64, leadHours []float64) (p10, p50, p90 []float64) {
	nSamples := len(leadHours)
	if m == nil || nSamples == 0 {
		return nil, nil, nil
	}
	p50s := m.PredictBatch(flatFeatures, nSamples)
	p10s := make([]float64, nSamples)
	p90s := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		if math.IsNaN(p50s[i]) || math.IsInf(p50s[i], 0) {
			p50s[i] = 0.0
		}
		bin := HorizonBinForLeadTimeWithBins(leadHours[i], m.HorizonBins)
		var off10, off90 float64
		if len(m.P10Offsets) > 0 {
			idx := bin
			if idx >= len(m.P10Offsets) {
				idx = len(m.P10Offsets) - 1
			}
			off10 = m.P10Offsets[idx]
		}
		if len(m.P90Offsets) > 0 {
			idx := bin
			if idx >= len(m.P90Offsets) {
				idx = len(m.P90Offsets) - 1
			}
			off90 = m.P90Offsets[idx]
		}

		if math.IsNaN(off10) || math.IsInf(off10, 0) {
			off10 = 0.0
		}
		if math.IsNaN(off90) || math.IsInf(off90, 0) {
			off90 = 0.0
		}

		p10s[i] = math.Min(p50s[i]+off10, p50s[i])
		p90s[i] = math.Max(p50s[i]+off90, p50s[i])
	}
	return p10s, p50s, p90s
}

// evaluateTreeFlat evaluates a tree for sampleIdx directly against flattened row-major features without allocations.
func evaluateTreeFlat(nodes []Node, features []float64, nFeatures int, sampleIdx int) float64 {
	if len(nodes) == 0 || sampleIdx < 0 || nFeatures <= 0 {
		return 0.0
	}
	baseOffset := sampleIdx * nFeatures
	curr := 0
	for steps := 0; steps < len(nodes); steps++ {
		if curr < 0 || curr >= len(nodes) {
			return 0.0
		}
		node := nodes[curr]
		if node.SplitFeature == -1 {
			return node.LeafValue
		}
		var fVal float64
		idx := baseOffset + node.SplitFeature
		if node.SplitFeature >= 0 && node.SplitFeature < nFeatures && idx >= 0 && idx < len(features) {
			fVal = features[idx]
		}
		if fVal < node.SplitValue {
			curr = node.LeftChild
		} else {
			curr = node.RightChild
		}
	}
	return 0.0
}

// FeatureVector represents an n-dimensional sample observation vector of floating-point features.
type FeatureVector struct {
	Values []float64
}

// Dataset represents a tabular dataset of feature vectors, target regression values, and optional sample weights.
type Dataset struct {
	X []FeatureVector
	Y []float64
	W []float64
}

// ColumnarDataset holds dataset features in a flattened Structure of Arrays (SoA) layout.
// Feature f for sample i is accessed at index f * NumSamples + i.
type ColumnarDataset struct {
	NumSamples  int
	NumFeatures int
	Features    []float64 // Size (NumFeatures * NumSamples)
	Targets     []float64 // Size NumSamples
	Weights     []float64 // Size NumSamples
	Binned      BinnedDataset
}

// ToColumnar converts legacy AoS feature vectors ([]FeatureVector) into flattened SoA ColumnarDataset.
func ToColumnar(X []FeatureVector, Y []float64, W []float64) ColumnarDataset {
	if len(X) == 0 {
		return ColumnarDataset{NumSamples: 0, NumFeatures: 0, Targets: Y, Weights: W}
	}
	nSamples := len(X)
	nFeatures := len(X[0].Values)
	features := make([]float64, nSamples*nFeatures)
	for f := 0; f < nFeatures; f++ {
		offset := f * nSamples
		for i := 0; i < nSamples; i++ {
			if f < len(X[i].Values) {
				features[offset+i] = X[i].Values[f]
			}
		}
	}
	cd := ColumnarDataset{
		NumSamples:  nSamples,
		NumFeatures: nFeatures,
		Features:    features,
		Targets:     Y,
		Weights:     W,
	}
	cd.Binned = QuantizeDataset(cd, 256)
	return cd
}

// EvaluateTree traverses a slice of decision tree nodes from root (node index 0) down to a leaf node
// for the given feature vector x, returning the leaf's prediction value.
func EvaluateTree(nodes []Node, x FeatureVector) float64 {
	if len(nodes) == 0 {
		return 0.0
	}
	curr := 0
	for steps := 0; steps < len(nodes); steps++ {
		if curr < 0 || curr >= len(nodes) {
			return 0.0
		}
		node := nodes[curr]
		if node.SplitFeature == -1 {
			return node.LeafValue
		}
		var fVal float64
		if node.SplitFeature >= 0 && node.SplitFeature < len(x.Values) {
			fVal = x.Values[node.SplitFeature]
		}
		if fVal < node.SplitValue {
			curr = node.LeftChild
		} else {
			curr = node.RightChild
		}
	}
	return 0.0
}

// EvaluateTreeColumnar evaluates a tree on sampleIdx within ColumnarDataset without allocations.
func EvaluateTreeColumnar(nodes []Node, data ColumnarDataset, sampleIdx int) float64 {
	if len(nodes) == 0 || sampleIdx < 0 || sampleIdx >= data.NumSamples {
		return 0.0
	}
	curr := 0
	for steps := 0; steps < len(nodes); steps++ {
		if curr < 0 || curr >= len(nodes) {
			return 0.0
		}
		node := nodes[curr]
		if node.SplitFeature == -1 {
			return node.LeafValue
		}
		var fVal float64
		idx := node.SplitFeature*data.NumSamples + sampleIdx
		if node.SplitFeature >= 0 && node.SplitFeature < data.NumFeatures && idx >= 0 && idx < len(data.Features) {
			fVal = data.Features[idx]
		}
		if fVal < node.SplitValue {
			curr = node.LeftChild
		} else {
			curr = node.RightChild
		}
	}
	return 0.0
}

// ScaleTreeLeaves returns a deep copy of tree with all leaf node values multiplied by factor.
func ScaleTreeLeaves(tree Tree, factor float64) Tree {
	nodes := make([]Node, len(tree.Nodes))
	copy(nodes, tree.Nodes)
	for i := range nodes {
		if nodes[i].SplitFeature == -1 {
			nodes[i].LeafValue *= factor
		}
	}
	return Tree{Nodes: nodes}
}

// Validate checks model invariants: finite BaseValue, valid LearningRate, and structural integrity of all trees.
func (m *Model) Validate() error {
	if m == nil {
		return errors.New("model is nil")
	}
	if math.IsNaN(m.BaseValue) || math.IsInf(m.BaseValue, 0) {
		return errors.New("model BaseValue is non-finite")
	}
	if math.IsNaN(m.LearningRate) || math.IsInf(m.LearningRate, 0) {
		return errors.New("model LearningRate is non-finite")
	}
	for tIdx, tree := range m.Trees {
		numNodes := len(tree.Nodes)
		if numNodes == 0 {
			continue
		}
		visited := make([]bool, numNodes)
		var checkNode func(idx, depth int) error
		checkNode = func(idx, depth int) error {
			if idx < 0 || idx >= numNodes {
				return fmt.Errorf("tree %d node index %d out of bounds [0, %d)", tIdx, idx, numNodes)
			}
			if depth > numNodes || depth > 128 {
				return fmt.Errorf("tree %d has cycle or depth exceeding limit at node %d", tIdx, idx)
			}
			if visited[idx] {
				return fmt.Errorf("tree %d contains cycle at node %d", tIdx, idx)
			}
			visited[idx] = true
			node := tree.Nodes[idx]
			if node.SplitFeature == -1 {
				if math.IsNaN(node.LeafValue) || math.IsInf(node.LeafValue, 0) {
					return fmt.Errorf("tree %d leaf node %d has non-finite LeafValue: %v", tIdx, idx, node.LeafValue)
				}
				return nil
			}
			if node.LeftChild < 0 || node.LeftChild >= numNodes {
				return fmt.Errorf("tree %d internal node %d has invalid LeftChild %d", tIdx, idx, node.LeftChild)
			}
			if node.RightChild < 0 || node.RightChild >= numNodes {
				return fmt.Errorf("tree %d internal node %d has invalid RightChild %d", tIdx, idx, node.RightChild)
			}
			if math.IsNaN(node.SplitValue) || math.IsInf(node.SplitValue, 0) {
				return fmt.Errorf("tree %d internal node %d has non-finite SplitValue", tIdx, idx)
			}
			if err := checkNode(node.LeftChild, depth+1); err != nil {
				return err
			}
			return checkNode(node.RightChild, depth+1)
		}
		if err := checkNode(0, 0); err != nil {
			return err
		}
	}
	return nil
}

// LoadModel loads a serialized GBDT model from a JSON file and validates its integrity.
func LoadModel(path string) (*Model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m Model
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	if err := m.Validate(); err != nil {
		return nil, fmt.Errorf("model validation failed: %w", err)
	}
	return &m, nil
}

// SaveModel serializes and writes the GBDT model to a JSON file.
func (m *Model) SaveModel(path string) error {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}
