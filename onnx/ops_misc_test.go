package onnx

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/onnx-gomlx/internal/protos"
)

func TestDropout(t *testing.T) {
	// Test that Dropout acts as identity in inference mode
	graphtest.RunTestGraphFn(t, "Dropout-inference", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}})

		node := &protos.NodeProto{
			OpType: "Dropout",
			Output: []string{"output"},
		}

		convertedOutputs := make(map[string]*Node)
		inputs = []*Node{x}
		result := convertDropout(convertedOutputs, node, inputs)
		outputs = []*Node{result}
		return
	}, []any{
		// In inference mode, dropout is identity
		[][]float32{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
	}, -1)

	// Test Dropout with mask output
	graphtest.RunTestGraphFn(t, "Dropout-with-mask", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{1.0, 2.0}, {3.0, 4.0}})

		node := &protos.NodeProto{
			OpType: "Dropout",
			Output: []string{"output", "mask"},
		}

		convertedOutputs := make(map[string]*Node)
		inputs = []*Node{x}
		result := convertDropout(convertedOutputs, node, inputs)
		mask := convertedOutputs["mask"]
		outputs = []*Node{result, mask}
		return
	}, []any{
		// Output is identity
		[][]float32{{1.0, 2.0}, {3.0, 4.0}},
		// Mask is all true
		[][]bool{{true, true}, {true, true}},
	}, -1)
}

func TestGlobalMaxPool(t *testing.T) {
	// Test GlobalMaxPool with 3D input [N, C, H]
	graphtest.RunTestGraphFn(t, "GlobalMaxPool-3D", func(g *Graph) (inputs, outputs []*Node) {
		// [1, 2, 4] - batch=1, channels=2, width=4
		x := Const(g, [][][]float32{{{1.0, 3.0, 2.0, 4.0}, {5.0, 7.0, 6.0, 8.0}}})

		inputs = []*Node{x}
		outputs = []*Node{convertGlobalMaxPool(inputs)}
		return
	}, []any{
		// Max over spatial dimension (axis 2): [[[4.0], [8.0]]]
		[][][]float32{{{4.0}, {8.0}}},
	}, -1)

	// Test GlobalMaxPool with 4D input [N, C, H, W]
	graphtest.RunTestGraphFn(t, "GlobalMaxPool-4D", func(g *Graph) (inputs, outputs []*Node) {
		// [1, 2, 2, 3] - batch=1, channels=2, height=2, width=3
		x := Const(g, [][][][]float32{{
			{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
			{{7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}},
		}})

		inputs = []*Node{x}
		outputs = []*Node{convertGlobalMaxPool(inputs)}
		return
	}, []any{
		// Max over spatial dimensions (axes 2, 3): [[[[6.0]], [[12.0]]]]
		[][][][]float32{{{{6.0}}, {{12.0}}}},
	}, -1)

	// Test GlobalMaxPool with batch size > 1
	graphtest.RunTestGraphFn(t, "GlobalMaxPool-batch", func(g *Graph) (inputs, outputs []*Node) {
		// [2, 1, 2, 2] - batch=2, channels=1, height=2, width=2
		x := Const(g, [][][][]float32{
			{{{1.0, 2.0}, {3.0, 4.0}}},
			{{{5.0, 6.0}, {7.0, 8.0}}},
		})

		inputs = []*Node{x}
		outputs = []*Node{convertGlobalMaxPool(inputs)}
		return
	}, []any{
		// Max for each batch independently
		[][][][]float32{{{{4.0}}}, {{{8.0}}}},
	}, -1)
}

func TestInstanceNormalization(t *testing.T) {
	// Test basic instance normalization
	graphtest.RunTestGraphFn(t, "InstanceNormalization-basic", func(g *Graph) (inputs, outputs []*Node) {
		// [1, 2, 3] - batch=1, channels=2, width=3
		x := Const(g, [][][]float32{{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}})
		scale := Const(g, []float32{1.0, 1.0})
		bias := Const(g, []float32{0.0, 0.0})

		node := &protos.NodeProto{
			OpType: "InstanceNormalization",
		}

		inputs = []*Node{x, scale, bias}
		outputs = []*Node{convertInstanceNormalization(node, inputs)}
		return
	}, []any{
		// Each channel normalized independently over spatial dimension
		// Channel 0: [1,2,3] -> mean=2, std≈0.8165 -> [-1.224, 0, 1.224]
		// Channel 1: [4,5,6] -> mean=5, std≈0.8165 -> [-1.224, 0, 1.224]
		[][][]float32{{{-1.2247, 0.0, 1.2247}, {-1.2247, 0.0, 1.2247}}},
	}, 1e-3)

	// Test with scale and bias
	graphtest.RunTestGraphFn(t, "InstanceNormalization-scale-bias", func(g *Graph) (inputs, outputs []*Node) {
		// [1, 2, 3] - batch=1, channels=2, width=3
		x := Const(g, [][][]float32{{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}})
		scale := Const(g, []float32{2.0, 0.5})
		bias := Const(g, []float32{1.0, -1.0})

		node := &protos.NodeProto{
			OpType: "InstanceNormalization",
		}

		inputs = []*Node{x, scale, bias}
		outputs = []*Node{convertInstanceNormalization(node, inputs)}
		return
	}, []any{
		// Channel 0: normalized * 2.0 + 1.0 = [-1.224*2+1, 0*2+1, 1.224*2+1]
		// Channel 1: normalized * 0.5 - 1.0 = [-1.224*0.5-1, 0*0.5-1, 1.224*0.5-1]
		[][][]float32{{{-1.4494, 1.0, 3.4494}, {-1.6124, -1.0, -0.3876}}},
	}, 1e-3)

	// Test 4D input [N, C, H, W]
	graphtest.RunTestGraphFn(t, "InstanceNormalization-4D", func(g *Graph) (inputs, outputs []*Node) {
		// [1, 2, 2, 2] - batch=1, channels=2, height=2, width=2
		x := Const(g, [][][][]float32{{
			{{1.0, 2.0}, {3.0, 4.0}},
			{{5.0, 6.0}, {7.0, 8.0}},
		}})
		scale := Const(g, []float32{1.0, 1.0})
		bias := Const(g, []float32{0.0, 0.0})

		node := &protos.NodeProto{
			OpType: "InstanceNormalization",
		}

		inputs = []*Node{x, scale, bias}
		outputs = []*Node{convertInstanceNormalization(node, inputs)}
		return
	}, []any{
		// Each channel normalized over H×W spatial dimensions
		// Channel 0: [1,2,3,4] -> mean=2.5, std≈1.118
		// Channel 1: [5,6,7,8] -> mean=6.5, std≈1.118
		[][][][]float32{{
			{{-1.3416, -0.4472}, {0.4472, 1.3416}},
			{{-1.3416, -0.4472}, {0.4472, 1.3416}},
		}},
	}, 1e-3)
}

func TestLpNormalization(t *testing.T) {
	// Test L1 normalization (p=1)
	graphtest.RunTestGraphFn(t, "LpNormalization-L1", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}})

		node := &protos.NodeProto{
			OpType: "LpNormalization",
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: 1},
				{Name: "p", Type: protos.AttributeProto_INT, I: 1},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{convertLpNormalization(node, inputs)}
		return
	}, []any{
		// L1 norm along axis 1:
		// Row 0: L1=6, normalized = [1/6, 2/6, 3/6] = [0.1667, 0.3333, 0.5]
		// Row 1: L1=15, normalized = [4/15, 5/15, 6/15] = [0.2667, 0.3333, 0.4]
		[][]float32{{0.1667, 0.3333, 0.5}, {0.2667, 0.3333, 0.4}},
	}, 1e-3)

	// Test L2 normalization (p=2, default)
	graphtest.RunTestGraphFn(t, "LpNormalization-L2", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{3.0, 4.0}, {5.0, 12.0}})

		node := &protos.NodeProto{
			OpType: "LpNormalization",
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: 1},
				{Name: "p", Type: protos.AttributeProto_INT, I: 2},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{convertLpNormalization(node, inputs)}
		return
	}, []any{
		// L2 norm along axis 1:
		// Row 0: L2=sqrt(9+16)=5, normalized = [3/5, 4/5] = [0.6, 0.8]
		// Row 1: L2=sqrt(25+144)=13, normalized = [5/13, 12/13]
		[][]float32{{0.6, 0.8}, {0.3846, 0.9231}},
	}, 1e-3)

	// Test with negative axis
	graphtest.RunTestGraphFn(t, "LpNormalization-negative-axis", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{3.0, 4.0}, {6.0, 8.0}})

		node := &protos.NodeProto{
			OpType: "LpNormalization",
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: -1}, // Last axis
				{Name: "p", Type: protos.AttributeProto_INT, I: 2},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{convertLpNormalization(node, inputs)}
		return
	}, []any{
		// Same as axis=1 for 2D tensor
		[][]float32{{0.6, 0.8}, {0.6, 0.8}},
	}, 1e-3)

	// Test normalization along axis 0
	graphtest.RunTestGraphFn(t, "LpNormalization-axis0", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{3.0, 0.0}, {4.0, 0.0}})

		node := &protos.NodeProto{
			OpType: "LpNormalization",
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: 0},
				{Name: "p", Type: protos.AttributeProto_INT, I: 2},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{convertLpNormalization(node, inputs)}
		return
	}, []any{
		// L2 norm along axis 0:
		// Column 0: L2=sqrt(9+16)=5, normalized = [3/5, 4/5]
		// Column 1: L2=0 (with epsilon protection)
		[][]float32{{0.6, 0.0}, {0.8, 0.0}},
	}, 1e-3)
}

func TestMeanVarianceNormalization(t *testing.T) {
	// Test basic mean-variance normalization with default axes
	graphtest.RunTestGraphFn(t, "MeanVarianceNormalization-default", func(g *Graph) (inputs, outputs []*Node) {
		// [2, 2, 2] input
		x := Const(g, [][][]float32{{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}})

		node := &protos.NodeProto{
			OpType: "MeanVarianceNormalization",
			// Default axes = [0, 2, 3], but for 3D it's effectively [0, 2]
		}

		inputs = []*Node{x}
		outputs = []*Node{convertMeanVarianceNormalization(node, inputs)}
		return
	}, []any{
		// Normalize over axes [0, 2, 3] (default)
		// For 3D tensor, this becomes [0, 2]
		// Mean across positions [0,:,0], [0,:,1], [1,:,0], [1,:,1] = (1+2+5+6)/4 = 3.5 for first channel
		// and (3+4+7+8)/4 = 5.5 for second channel
		// But the default axes [0,2,3] on a 3D tensor will normalize differently
		[][][]float32{{{-1.5275, -1.0911}, {-0.6547, -0.2182}}, {{0.2182, 0.6547}, {1.0911, 1.5275}}},
	}, 1e-3)

	// Test with custom axes
	graphtest.RunTestGraphFn(t, "MeanVarianceNormalization-custom-axes", func(g *Graph) (inputs, outputs []*Node) {
		// [2, 3] input
		x := Const(g, [][]float32{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}})

		node := &protos.NodeProto{
			OpType: "MeanVarianceNormalization",
			Attribute: []*protos.AttributeProto{
				{Name: "axes", Type: protos.AttributeProto_INTS, Ints: []int64{1}},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{convertMeanVarianceNormalization(node, inputs)}
		return
	}, []any{
		// Normalize each row independently (axis 1)
		// Row 0: mean=2, std≈0.8165 -> [-1.224, 0, 1.224]
		// Row 1: mean=5, std≈0.8165 -> [-1.224, 0, 1.224]
		[][]float32{{-1.2247, 0.0, 1.2247}, {-1.2247, 0.0, 1.2247}},
	}, 1e-3)

	// Test normalization to zero mean and unit variance
	graphtest.RunTestGraphFn(t, "MeanVarianceNormalization-verify-stats", func(g *Graph) (inputs, outputs []*Node) {
		// Simple 1D-like data
		x := Const(g, [][]float32{{10.0, 20.0, 30.0, 40.0, 50.0}})

		node := &protos.NodeProto{
			OpType: "MeanVarianceNormalization",
			Attribute: []*protos.AttributeProto{
				{Name: "axes", Type: protos.AttributeProto_INTS, Ints: []int64{1}},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{convertMeanVarianceNormalization(node, inputs)}
		return
	}, []any{
		// Mean = 30, std ≈ 14.142
		// Normalized: [-1.414, -0.707, 0, 0.707, 1.414]
		[][]float32{{-1.4142, -0.7071, 0.0, 0.7071, 1.4142}},
	}, 1e-3)
}

func TestCastLike(t *testing.T) {
	// Test casting float32 to float64
	graphtest.RunTestGraphFn(t, "CastLike-float32-to-float64", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{1.5, 2.5, 3.5})
		target := Const(g, []float64{0.0}) // Target dtype is float64

		inputs = []*Node{x, target}
		outputs = []*Node{convertCastLike(inputs)}
		return
	}, []any{
		[]float64{1.5, 2.5, 3.5},
	}, -1)

	// Test casting int32 to float32
	graphtest.RunTestGraphFn(t, "CastLike-int32-to-float32", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []int32{1, 2, 3})
		target := Const(g, float32(0.0))

		inputs = []*Node{x, target}
		outputs = []*Node{convertCastLike(inputs)}
		return
	}, []any{
		[]float32{1.0, 2.0, 3.0},
	}, -1)

	// Test casting float32 to int32
	graphtest.RunTestGraphFn(t, "CastLike-float32-to-int32", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{1.9, 2.1, 3.7})
		target := Const(g, int32(0))

		inputs = []*Node{x, target}
		outputs = []*Node{convertCastLike(inputs)}
		return
	}, []any{
		// Truncation towards zero
		[]int32{1, 2, 3},
	}, -1)

	// Test casting bool to int32
	graphtest.RunTestGraphFn(t, "CastLike-bool-to-int32", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []bool{true, false, true})
		target := Const(g, int32(0))

		inputs = []*Node{x, target}
		outputs = []*Node{convertCastLike(inputs)}
		return
	}, []any{
		[]int32{1, 0, 1},
	}, -1)

	// Test casting with 2D tensors
	graphtest.RunTestGraphFn(t, "CastLike-2D", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]int64{{1, 2}, {3, 4}})
		target := Const(g, [][]float32{{0.0}})

		inputs = []*Node{x, target}
		outputs = []*Node{convertCastLike(inputs)}
		return
	}, []any{
		[][]float32{{1.0, 2.0}, {3.0, 4.0}},
	}, -1)

	// Test no-op cast (same dtype)
	graphtest.RunTestGraphFn(t, "CastLike-no-op", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{1.0, 2.0, 3.0})
		target := Const(g, []float32{0.0})

		inputs = []*Node{x, target}
		outputs = []*Node{convertCastLike(inputs)}
		return
	}, []any{
		[]float32{1.0, 2.0, 3.0},
	}, -1)
}
