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
	"math"
	"os"
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
	if m == nil || nSamples <= 0 || len(features) == 0 {
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

// evaluateTreeFlat evaluates a tree for sampleIdx directly against flattened row-major features without allocations.
func evaluateTreeFlat(nodes []Node, features []float64, nFeatures int, sampleIdx int) float64 {
	if len(nodes) == 0 {
		return 0.0
	}
	baseOffset := sampleIdx * nFeatures
	curr := 0
	for {
		if curr < 0 || curr >= len(nodes) {
			return 0.0
		}
		node := nodes[curr]
		if node.SplitFeature == -1 {
			return node.LeafValue
		}
		var fVal float64
		if node.SplitFeature >= 0 && node.SplitFeature < nFeatures {
			fVal = features[baseOffset+node.SplitFeature]
		}
		if fVal < node.SplitValue {
			curr = node.LeftChild
		} else {
			curr = node.RightChild
		}
	}
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
	for {
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
}

// EvaluateTreeColumnar evaluates a tree on sampleIdx within ColumnarDataset without allocations.
func EvaluateTreeColumnar(nodes []Node, data ColumnarDataset, sampleIdx int) float64 {
	if len(nodes) == 0 {
		return 0.0
	}
	curr := 0
	for {
		if curr < 0 || curr >= len(nodes) {
			return 0.0
		}
		node := nodes[curr]
		if node.SplitFeature == -1 {
			return node.LeafValue
		}
		var fVal float64
		if node.SplitFeature >= 0 && node.SplitFeature < data.NumFeatures {
			fVal = data.Features[node.SplitFeature*data.NumSamples+sampleIdx]
		}
		if fVal < node.SplitValue {
			curr = node.LeftChild
		} else {
			curr = node.RightChild
		}
	}
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

// LoadModel loads a serialized GBDT model from a JSON file.
func LoadModel(path string) (*Model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m Model
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
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
