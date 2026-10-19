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

package treeboost

import (
	"math"
	"testing"
)

func TestPartitionSamplesInPlace_NaNSafety(t *testing.T) {
	sampleIdxs := []int{0, 1, 2, 3}
	featureValues := []float64{math.NaN(), 10.0, math.NaN(), 2.0}
	splitVal := 5.0

	mid := partitionSamplesInPlace(sampleIdxs, 0, 4, featureValues, 0, splitVal)
	// Left of mid must be strictly < splitVal and not NaN
	for i := 0; i < mid; i++ {
		v := featureValues[sampleIdxs[i]]
		if math.IsNaN(v) || v >= splitVal {
			t.Errorf("sample at left %d has value %f, expected < %f", i, v, splitVal)
		}
	}
	// Right of mid must be >= splitVal or NaN
	for i := mid; i < 4; i++ {
		v := featureValues[sampleIdxs[i]]
		if !math.IsNaN(v) && v < splitVal {
			t.Errorf("sample at right %d has value %f, expected >= %f or NaN", i, v, splitVal)
		}
	}
}
