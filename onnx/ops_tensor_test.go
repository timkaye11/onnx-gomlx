package onnx

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/onnx-gomlx/internal/protos"
)

func TestSize(t *testing.T) {
	// Test Size with 1D tensor
	graphtest.RunTestGraphFn(t, "Size-1D", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{1, 2, 3, 4, 5})
		inputs = []*Node{x}
		outputs = []*Node{convertSize(inputs)}
		return
	}, []any{
		int64(5),
	}, -1)

	// Test Size with 2D tensor
	graphtest.RunTestGraphFn(t, "Size-2D", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{1, 2, 3}, {4, 5, 6}})
		inputs = []*Node{x}
		outputs = []*Node{convertSize(inputs)}
		return
	}, []any{
		int64(6),
	}, -1)

	// Test Size with 3D tensor
	graphtest.RunTestGraphFn(t, "Size-3D", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][][]float32{
			{{1, 2}, {3, 4}},
			{{5, 6}, {7, 8}},
		})
		inputs = []*Node{x}
		outputs = []*Node{convertSize(inputs)}
		return
	}, []any{
		int64(8),
	}, -1)

	// Test Size with scalar
	graphtest.RunTestGraphFn(t, "Size-scalar", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, float32(42))
		inputs = []*Node{x}
		outputs = []*Node{convertSize(inputs)}
		return
	}, []any{
		int64(1),
	}, -1)
}

func TestOneHot(t *testing.T) {
	// Test OneHot with default axis (-1)
	graphtest.RunTestGraphFn(t, "OneHot-axis-neg1", func(g *Graph) (inputs, outputs []*Node) {
		indices := Const(g, []int32{0, 2, 1})
		depth := Const(g, int64(3))
		values := Const(g, []float32{0.0, 1.0})

		node := &protos.NodeProto{
			OpType: "OneHot",
			Input:  []string{"indices", "depth", "values"},
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: -1},
			},
		}

		model := &Model{
			variableNameToValue: make(map[string]*protos.TensorProto),
			nodeOutputToNode:    make(map[string]*protos.NodeProto),
		}

		convertedOutputs := make(map[string]*Node)
		convertedOutputs["depth"] = depth

		inputs = []*Node{indices, depth, values}
		outputs = []*Node{convertOneHot(model, convertedOutputs, node, inputs)}
		return
	}, []any{
		[][]float32{{1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}},
	}, -1)

	// Test OneHot with axis=0
	graphtest.RunTestGraphFn(t, "OneHot-axis-0", func(g *Graph) (inputs, outputs []*Node) {
		indices := Const(g, []int32{0, 2, 1})
		depth := Const(g, int64(3))
		values := Const(g, []float32{0.0, 1.0})

		node := &protos.NodeProto{
			OpType: "OneHot",
			Input:  []string{"indices", "depth", "values"},
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		model := &Model{
			variableNameToValue: make(map[string]*protos.TensorProto),
			nodeOutputToNode:    make(map[string]*protos.NodeProto),
		}

		convertedOutputs := make(map[string]*Node)
		convertedOutputs["depth"] = depth

		inputs = []*Node{indices, depth, values}
		outputs = []*Node{convertOneHot(model, convertedOutputs, node, inputs)}
		return
	}, []any{
		[][]float32{{1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}},
	}, -1)

	// Test OneHot with custom on/off values
	graphtest.RunTestGraphFn(t, "OneHot-custom-values", func(g *Graph) (inputs, outputs []*Node) {
		indices := Const(g, []int32{0, 1})
		depth := Const(g, int64(2))
		values := Const(g, []float32{-1.0, 10.0})

		node := &protos.NodeProto{
			OpType: "OneHot",
			Input:  []string{"indices", "depth", "values"},
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: -1},
			},
		}

		model := &Model{
			variableNameToValue: make(map[string]*protos.TensorProto),
			nodeOutputToNode:    make(map[string]*protos.NodeProto),
		}

		convertedOutputs := make(map[string]*Node)
		convertedOutputs["depth"] = depth

		inputs = []*Node{indices, depth, values}
		outputs = []*Node{convertOneHot(model, convertedOutputs, node, inputs)}
		return
	}, []any{
		[][]float32{{10.0, -1.0}, {-1.0, 10.0}},
	}, -1)

	// Test OneHot with 2D indices
	graphtest.RunTestGraphFn(t, "OneHot-2D-indices", func(g *Graph) (inputs, outputs []*Node) {
		indices := Const(g, [][]int32{{0, 1}, {1, 0}})
		depth := Const(g, int64(2))
		values := Const(g, []float32{0.0, 1.0})

		node := &protos.NodeProto{
			OpType: "OneHot",
			Input:  []string{"indices", "depth", "values"},
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: -1},
			},
		}

		model := &Model{
			variableNameToValue: make(map[string]*protos.TensorProto),
			nodeOutputToNode:    make(map[string]*protos.NodeProto),
		}

		convertedOutputs := make(map[string]*Node)
		convertedOutputs["depth"] = depth

		inputs = []*Node{indices, depth, values}
		outputs = []*Node{convertOneHot(model, convertedOutputs, node, inputs)}
		return
	}, []any{
		[][][]float32{
			{{1.0, 0.0}, {0.0, 1.0}},
			{{0.0, 1.0}, {1.0, 0.0}},
		},
	}, -1)
}

func TestEyeLike(t *testing.T) {
	// Test EyeLike with k=0 (main diagonal)
	graphtest.RunTestGraphFn(t, "EyeLike-k0", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}})

		node := &protos.NodeProto{
			OpType: "EyeLike",
			Attribute: []*protos.AttributeProto{
				{Name: "k", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{convertEyeLike(node, inputs)}
		return
	}, []any{
		[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
	}, -1)

	// Test EyeLike with k=1 (upper diagonal)
	graphtest.RunTestGraphFn(t, "EyeLike-k1", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}})

		node := &protos.NodeProto{
			OpType: "EyeLike",
			Attribute: []*protos.AttributeProto{
				{Name: "k", Type: protos.AttributeProto_INT, I: 1},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{convertEyeLike(node, inputs)}
		return
	}, []any{
		[][]float32{{0, 1, 0}, {0, 0, 1}, {0, 0, 0}},
	}, -1)

	// Test EyeLike with k=-1 (lower diagonal)
	graphtest.RunTestGraphFn(t, "EyeLike-k-neg1", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}})

		node := &protos.NodeProto{
			OpType: "EyeLike",
			Attribute: []*protos.AttributeProto{
				{Name: "k", Type: protos.AttributeProto_INT, I: -1},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{convertEyeLike(node, inputs)}
		return
	}, []any{
		[][]float32{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}},
	}, -1)

	// Test EyeLike with non-square matrix
	graphtest.RunTestGraphFn(t, "EyeLike-non-square", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}})

		node := &protos.NodeProto{
			OpType: "EyeLike",
			Attribute: []*protos.AttributeProto{
				{Name: "k", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{convertEyeLike(node, inputs)}
		return
	}, []any{
		[][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}},
	}, -1)

	// Test EyeLike with different dtype
	graphtest.RunTestGraphFn(t, "EyeLike-dtype-int32", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{0, 0}, {0, 0}})

		node := &protos.NodeProto{
			OpType: "EyeLike",
			Attribute: []*protos.AttributeProto{
				{Name: "k", Type: protos.AttributeProto_INT, I: 0},
				{Name: "dtype", Type: protos.AttributeProto_INT, I: int64(protos.TensorProto_INT32)},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{convertEyeLike(node, inputs)}
		return
	}, []any{
		[][]int32{{1, 0}, {0, 1}},
	}, -1)
}

func TestDepthToSpace(t *testing.T) {
	// Test DepthToSpace with DCR mode
	graphtest.RunTestGraphFn(t, "DepthToSpace-DCR", func(g *Graph) (inputs, outputs []*Node) {
		// Input: [1, 8, 2, 2] -> Output: [1, 2, 4, 4]
		// blocksize = 2, so C reduces by 4 (2x2), H and W each double
		x := Const(g, [][][][]float32{{
			{{1, 2}, {3, 4}},     // channel 0
			{{5, 6}, {7, 8}},     // channel 1
			{{9, 10}, {11, 12}},  // channel 2
			{{13, 14}, {15, 16}}, // channel 3
			{{17, 18}, {19, 20}}, // channel 4
			{{21, 22}, {23, 24}}, // channel 5
			{{25, 26}, {27, 28}}, // channel 6
			{{29, 30}, {31, 32}}, // channel 7
		}})

		node := &protos.NodeProto{
			OpType: "DepthToSpace",
			Attribute: []*protos.AttributeProto{
				{Name: "blocksize", Type: protos.AttributeProto_INT, I: 2},
				{Name: "mode", Type: protos.AttributeProto_STRING, S: []byte("DCR")},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{convertDepthToSpace(node, inputs)}
		return
	}, []any{
		[][][][]float32{{
			{
				{1, 5, 2, 6},
				{9, 13, 10, 14},
				{3, 7, 4, 8},
				{11, 15, 12, 16},
			},
			{
				{17, 21, 18, 22},
				{25, 29, 26, 30},
				{19, 23, 20, 24},
				{27, 31, 28, 32},
			},
		}},
	}, -1)

	// Test DepthToSpace with CRD mode
	graphtest.RunTestGraphFn(t, "DepthToSpace-CRD", func(g *Graph) (inputs, outputs []*Node) {
		// Input: [1, 8, 2, 2] -> Output: [1, 2, 4, 4]
		x := Const(g, [][][][]float32{{
			{{1, 2}, {3, 4}},     // channel 0
			{{5, 6}, {7, 8}},     // channel 1
			{{9, 10}, {11, 12}},  // channel 2
			{{13, 14}, {15, 16}}, // channel 3
			{{17, 18}, {19, 20}}, // channel 4
			{{21, 22}, {23, 24}}, // channel 5
			{{25, 26}, {27, 28}}, // channel 6
			{{29, 30}, {31, 32}}, // channel 7
		}})

		node := &protos.NodeProto{
			OpType: "DepthToSpace",
			Attribute: []*protos.AttributeProto{
				{Name: "blocksize", Type: protos.AttributeProto_INT, I: 2},
				{Name: "mode", Type: protos.AttributeProto_STRING, S: []byte("CRD")},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{convertDepthToSpace(node, inputs)}
		return
	}, []any{
		[][][][]float32{{
			{
				{1, 2, 5, 6},
				{3, 4, 7, 8},
				{9, 10, 13, 14},
				{11, 12, 15, 16},
			},
			{
				{17, 18, 21, 22},
				{19, 20, 23, 24},
				{25, 26, 29, 30},
				{27, 28, 31, 32},
			},
		}},
	}, -1)

	// Test DepthToSpace with simpler case
	graphtest.RunTestGraphFn(t, "DepthToSpace-simple", func(g *Graph) (inputs, outputs []*Node) {
		// Input: [1, 4, 1, 1] -> Output: [1, 1, 2, 2]
		x := Const(g, [][][][]float32{{
			{{1}},
			{{2}},
			{{3}},
			{{4}},
		}})

		node := &protos.NodeProto{
			OpType: "DepthToSpace",
			Attribute: []*protos.AttributeProto{
				{Name: "blocksize", Type: protos.AttributeProto_INT, I: 2},
				{Name: "mode", Type: protos.AttributeProto_STRING, S: []byte("DCR")},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{convertDepthToSpace(node, inputs)}
		return
	}, []any{
		[][][][]float32{{
			{
				{1, 2},
				{3, 4},
			},
		}},
	}, -1)
}

func TestSpaceToDepth(t *testing.T) {
	// Test SpaceToDepth basic
	graphtest.RunTestGraphFn(t, "SpaceToDepth-basic", func(g *Graph) (inputs, outputs []*Node) {
		// Input: [1, 1, 2, 2] -> Output: [1, 4, 1, 1]
		// blocksize = 2, so spatial dims reduce by 2, depth increases by 4
		x := Const(g, [][][][]float32{{
			{
				{1, 2},
				{3, 4},
			},
		}})

		node := &protos.NodeProto{
			OpType: "SpaceToDepth",
			Attribute: []*protos.AttributeProto{
				{Name: "blocksize", Type: protos.AttributeProto_INT, I: 2},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{convertSpaceToDepth(node, inputs)}
		return
	}, []any{
		[][][][]float32{{
			{{1}},
			{{2}},
			{{3}},
			{{4}},
		}},
	}, -1)

	// Test SpaceToDepth larger input
	graphtest.RunTestGraphFn(t, "SpaceToDepth-larger", func(g *Graph) (inputs, outputs []*Node) {
		// Input: [1, 2, 4, 4] -> Output: [1, 8, 2, 2]
		x := Const(g, [][][][]float32{{
			{
				{1, 5, 2, 6},
				{9, 13, 10, 14},
				{3, 7, 4, 8},
				{11, 15, 12, 16},
			},
			{
				{17, 21, 18, 22},
				{25, 29, 26, 30},
				{19, 23, 20, 24},
				{27, 31, 28, 32},
			},
		}})

		node := &protos.NodeProto{
			OpType: "SpaceToDepth",
			Attribute: []*protos.AttributeProto{
				{Name: "blocksize", Type: protos.AttributeProto_INT, I: 2},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{convertSpaceToDepth(node, inputs)}
		return
	}, []any{
		[][][][]float32{{
			{{1, 2}, {3, 4}},
			{{5, 6}, {7, 8}},
			{{9, 10}, {11, 12}},
			{{13, 14}, {15, 16}},
			{{17, 18}, {19, 20}},
			{{21, 22}, {23, 24}},
			{{25, 26}, {27, 28}},
			{{29, 30}, {31, 32}},
		}},
	}, -1)

	// Test SpaceToDepth with multiple channels
	graphtest.RunTestGraphFn(t, "SpaceToDepth-multi-channel", func(g *Graph) (inputs, outputs []*Node) {
		// Input: [1, 2, 2, 2] -> Output: [1, 8, 1, 1]
		x := Const(g, [][][][]float32{{
			{{1, 2}, {3, 4}},
			{{5, 6}, {7, 8}},
		}})

		node := &protos.NodeProto{
			OpType: "SpaceToDepth",
			Attribute: []*protos.AttributeProto{
				{Name: "blocksize", Type: protos.AttributeProto_INT, I: 2},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{convertSpaceToDepth(node, inputs)}
		return
	}, []any{
		[][][][]float32{{
			{{1}},
			{{2}},
			{{3}},
			{{4}},
			{{5}},
			{{6}},
			{{7}},
			{{8}},
		}},
	}, -1)
}
