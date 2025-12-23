package onnx

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/onnx-gomlx/internal/protos"
)

func TestReduceMax(t *testing.T) {
	// Test ReduceMax with keepdims=true along axis 0
	graphtest.RunTestGraphFn(t, "ReduceMax-keepdims-axis0", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}})

		node := &protos.NodeProto{
			OpType: "ReduceMax",
			Attribute: []*protos.AttributeProto{
				{Name: "axes", Type: protos.AttributeProto_INTS, Ints: []int64{0}},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 1},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertReduceMax(nil, nil, node, inputs),
		}
		return
	}, []any{
		// Max along axis 0: [[3.0, 6.0]] (keepdims=true keeps rank 2)
		[][]float32{{3.0, 6.0}},
	}, -1)

	// Test ReduceMax with keepdims=false along axis 0
	graphtest.RunTestGraphFn(t, "ReduceMax-no-keepdims-axis0", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}})

		node := &protos.NodeProto{
			OpType: "ReduceMax",
			Attribute: []*protos.AttributeProto{
				{Name: "axes", Type: protos.AttributeProto_INTS, Ints: []int64{0}},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertReduceMax(nil, nil, node, inputs),
		}
		return
	}, []any{
		// Max along axis 0: [3.0, 6.0] (keepdims=false reduces rank)
		[]float32{3.0, 6.0},
	}, -1)

	// Test ReduceMax along axis 1
	graphtest.RunTestGraphFn(t, "ReduceMax-axis1", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{1.0, 4.0, 2.0}, {5.0, 2.0, 6.0}, {3.0, 7.0, 1.0}})

		node := &protos.NodeProto{
			OpType: "ReduceMax",
			Attribute: []*protos.AttributeProto{
				{Name: "axes", Type: protos.AttributeProto_INTS, Ints: []int64{1}},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertReduceMax(nil, nil, node, inputs),
		}
		return
	}, []any{
		// Max along axis 1: [4.0, 6.0, 7.0]
		[]float32{4.0, 6.0, 7.0},
	}, -1)

	// Test ReduceMax with all axes (reduce to scalar)
	graphtest.RunTestGraphFn(t, "ReduceMax-all-axes", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}})

		node := &protos.NodeProto{
			OpType: "ReduceMax",
			Attribute: []*protos.AttributeProto{
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertReduceMax(nil, nil, node, inputs),
		}
		return
	}, []any{
		// Max of all elements: 6.0
		float32(6.0),
	}, -1)
}

func TestReduceMin(t *testing.T) {
	// Test ReduceMin with keepdims=true along axis 0
	graphtest.RunTestGraphFn(t, "ReduceMin-keepdims-axis0", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{3.0, 6.0}, {2.0, 5.0}, {1.0, 4.0}})

		node := &protos.NodeProto{
			OpType: "ReduceMin",
			Attribute: []*protos.AttributeProto{
				{Name: "axes", Type: protos.AttributeProto_INTS, Ints: []int64{0}},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 1},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertReduceMin(nil, nil, node, inputs),
		}
		return
	}, []any{
		// Min along axis 0: [[1.0, 4.0]]
		[][]float32{{1.0, 4.0}},
	}, -1)

	// Test ReduceMin along axis 1
	graphtest.RunTestGraphFn(t, "ReduceMin-axis1", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{5.0, 2.0, 8.0}, {1.0, 9.0, 3.0}, {7.0, 4.0, 6.0}})

		node := &protos.NodeProto{
			OpType: "ReduceMin",
			Attribute: []*protos.AttributeProto{
				{Name: "axes", Type: protos.AttributeProto_INTS, Ints: []int64{1}},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertReduceMin(nil, nil, node, inputs),
		}
		return
	}, []any{
		// Min along axis 1: [2.0, 1.0, 4.0]
		[]float32{2.0, 1.0, 4.0},
	}, -1)

	// Test ReduceMin with all axes
	graphtest.RunTestGraphFn(t, "ReduceMin-all-axes", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{5.0, 2.0}, {8.0, 1.0}, {9.0, 3.0}})

		node := &protos.NodeProto{
			OpType: "ReduceMin",
			Attribute: []*protos.AttributeProto{
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertReduceMin(nil, nil, node, inputs),
		}
		return
	}, []any{
		// Min of all elements: 1.0
		float32(1.0),
	}, -1)
}

func TestReduceSum(t *testing.T) {
	// Test ReduceSum with keepdims=true along axis 0
	graphtest.RunTestGraphFn(t, "ReduceSum-keepdims-axis0", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}})

		node := &protos.NodeProto{
			OpType: "ReduceSum",
			Attribute: []*protos.AttributeProto{
				{Name: "axes", Type: protos.AttributeProto_INTS, Ints: []int64{0}},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 1},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertReduceSum(nil, nil, node, inputs),
		}
		return
	}, []any{
		// Sum along axis 0: [[9.0, 12.0]] (1+3+5=9, 2+4+6=12)
		[][]float32{{9.0, 12.0}},
	}, -1)

	// Test ReduceSum along axis 1
	graphtest.RunTestGraphFn(t, "ReduceSum-axis1", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}})

		node := &protos.NodeProto{
			OpType: "ReduceSum",
			Attribute: []*protos.AttributeProto{
				{Name: "axes", Type: protos.AttributeProto_INTS, Ints: []int64{1}},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertReduceSum(nil, nil, node, inputs),
		}
		return
	}, []any{
		// Sum along axis 1: [6.0, 15.0] (1+2+3=6, 4+5+6=15)
		[]float32{6.0, 15.0},
	}, -1)

	// Test ReduceSum with all axes
	graphtest.RunTestGraphFn(t, "ReduceSum-all-axes", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{1.0, 2.0}, {3.0, 4.0}})

		node := &protos.NodeProto{
			OpType: "ReduceSum",
			Attribute: []*protos.AttributeProto{
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertReduceSum(nil, nil, node, inputs),
		}
		return
	}, []any{
		// Sum of all elements: 10.0
		float32(10.0),
	}, -1)
}

func TestReduceProd(t *testing.T) {
	// Test ReduceProd with keepdims=true along axis 0
	graphtest.RunTestGraphFn(t, "ReduceProd-keepdims-axis0", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{2.0, 3.0}, {4.0, 5.0}})

		node := &protos.NodeProto{
			OpType: "ReduceProd",
			Attribute: []*protos.AttributeProto{
				{Name: "axes", Type: protos.AttributeProto_INTS, Ints: []int64{0}},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 1},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertReduceProd(nil, nil, node, inputs),
		}
		return
	}, []any{
		// Product along axis 0: [[8.0, 15.0]] (2*4=8, 3*5=15)
		[][]float32{{8.0, 15.0}},
	}, -1)

	// Test ReduceProd along axis 1
	graphtest.RunTestGraphFn(t, "ReduceProd-axis1", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{2.0, 3.0, 4.0}, {1.0, 2.0, 3.0}})

		node := &protos.NodeProto{
			OpType: "ReduceProd",
			Attribute: []*protos.AttributeProto{
				{Name: "axes", Type: protos.AttributeProto_INTS, Ints: []int64{1}},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertReduceProd(nil, nil, node, inputs),
		}
		return
	}, []any{
		// Product along axis 1: [24.0, 6.0] (2*3*4=24, 1*2*3=6)
		[]float32{24.0, 6.0},
	}, -1)

	// Test ReduceProd with all axes
	graphtest.RunTestGraphFn(t, "ReduceProd-all-axes", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{2.0, 3.0}, {4.0, 5.0}})

		node := &protos.NodeProto{
			OpType: "ReduceProd",
			Attribute: []*protos.AttributeProto{
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertReduceProd(nil, nil, node, inputs),
		}
		return
	}, []any{
		// Product of all elements: 120.0 (2*3*4*5=120)
		float32(120.0),
	}, -1)
}

func TestReduceL1(t *testing.T) {
	// Test ReduceL1 (sum of absolute values) along axis 0
	graphtest.RunTestGraphFn(t, "ReduceL1-axis0", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{-1.0, 2.0}, {-3.0, 4.0}, {5.0, -6.0}})

		node := &protos.NodeProto{
			OpType: "ReduceL1",
			Attribute: []*protos.AttributeProto{
				{Name: "axes", Type: protos.AttributeProto_INTS, Ints: []int64{0}},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertReduceL1(nil, nil, node, inputs),
		}
		return
	}, []any{
		// L1 along axis 0: [9.0, 12.0] (|-1|+|-3|+|5|=9, |2|+|4|+|-6|=12)
		[]float32{9.0, 12.0},
	}, -1)

	// Test ReduceL1 along axis 1 with keepdims
	graphtest.RunTestGraphFn(t, "ReduceL1-axis1-keepdims", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{-1.0, -2.0, 3.0}, {4.0, -5.0, -6.0}})

		node := &protos.NodeProto{
			OpType: "ReduceL1",
			Attribute: []*protos.AttributeProto{
				{Name: "axes", Type: protos.AttributeProto_INTS, Ints: []int64{1}},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 1},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertReduceL1(nil, nil, node, inputs),
		}
		return
	}, []any{
		// L1 along axis 1: [[6.0], [15.0]] (|-1|+|-2|+|3|=6, |4|+|-5|+|-6|=15)
		[][]float32{{6.0}, {15.0}},
	}, -1)

	// Test ReduceL1 with all axes
	graphtest.RunTestGraphFn(t, "ReduceL1-all-axes", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{-1.0, 2.0}, {-3.0, 4.0}})

		node := &protos.NodeProto{
			OpType: "ReduceL1",
			Attribute: []*protos.AttributeProto{
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertReduceL1(nil, nil, node, inputs),
		}
		return
	}, []any{
		// L1 of all elements: 10.0 (|-1|+|2|+|-3|+|4|=10)
		float32(10.0),
	}, -1)
}

func TestReduceL2(t *testing.T) {
	// Test ReduceL2 (Euclidean norm) along axis 0
	graphtest.RunTestGraphFn(t, "ReduceL2-axis0", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{3.0, 4.0}, {0.0, 0.0}})

		node := &protos.NodeProto{
			OpType: "ReduceL2",
			Attribute: []*protos.AttributeProto{
				{Name: "axes", Type: protos.AttributeProto_INTS, Ints: []int64{0}},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertReduceL2(nil, nil, node, inputs),
		}
		return
	}, []any{
		// L2 along axis 0: [3.0, 4.0] (sqrt(3^2+0^2)=3, sqrt(4^2+0^2)=4)
		[]float32{3.0, 4.0},
	}, -1)

	// Test ReduceL2 along axis 1
	graphtest.RunTestGraphFn(t, "ReduceL2-axis1", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{3.0, 4.0}, {5.0, 12.0}})

		node := &protos.NodeProto{
			OpType: "ReduceL2",
			Attribute: []*protos.AttributeProto{
				{Name: "axes", Type: protos.AttributeProto_INTS, Ints: []int64{1}},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertReduceL2(nil, nil, node, inputs),
		}
		return
	}, []any{
		// L2 along axis 1: [5.0, 13.0] (sqrt(3^2+4^2)=5, sqrt(5^2+12^2)=13)
		[]float32{5.0, 13.0},
	}, -1)

	// Test ReduceL2 with keepdims
	graphtest.RunTestGraphFn(t, "ReduceL2-keepdims", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{1.0, 2.0}, {2.0, 4.0}})

		node := &protos.NodeProto{
			OpType: "ReduceL2",
			Attribute: []*protos.AttributeProto{
				{Name: "axes", Type: protos.AttributeProto_INTS, Ints: []int64{0}},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 1},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertReduceL2(nil, nil, node, inputs),
		}
		return
	}, []any{
		// L2 along axis 0 with keepdims: [[2.236, 4.472]]
		// sqrt(1^2+2^2)=sqrt(5)≈2.236, sqrt(2^2+4^2)=sqrt(20)≈4.472
		[][]float32{{2.236, 4.472}},
	}, 1e-3)

	// Test ReduceL2 with all axes
	graphtest.RunTestGraphFn(t, "ReduceL2-all-axes", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{3.0, 4.0}, {0.0, 0.0}})

		node := &protos.NodeProto{
			OpType: "ReduceL2",
			Attribute: []*protos.AttributeProto{
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertReduceL2(nil, nil, node, inputs),
		}
		return
	}, []any{
		// L2 of all elements: 5.0 (sqrt(3^2+4^2+0^2+0^2)=5)
		float32(5.0),
	}, -1)
}

func TestArgMax(t *testing.T) {
	// Test ArgMax along axis 0 with keepdims=true
	graphtest.RunTestGraphFn(t, "ArgMax-axis0-keepdims", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{1.0, 4.0}, {3.0, 2.0}, {2.0, 5.0}})

		node := &protos.NodeProto{
			OpType: "ArgMax",
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: 0},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 1},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertArgMax(node, inputs),
		}
		return
	}, []any{
		// ArgMax along axis 0: [[1, 2]] (indices where max occurs)
		// Column 0: max is 3.0 at index 1
		// Column 1: max is 5.0 at index 2
		[][]int64{{1, 2}},
	}, -1)

	// Test ArgMax along axis 1 without keepdims
	graphtest.RunTestGraphFn(t, "ArgMax-axis1-no-keepdims", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{1.0, 4.0, 2.0}, {5.0, 2.0, 6.0}, {3.0, 7.0, 1.0}})

		node := &protos.NodeProto{
			OpType: "ArgMax",
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: 1},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertArgMax(node, inputs),
		}
		return
	}, []any{
		// ArgMax along axis 1: [1, 2, 1]
		// Row 0: max is 4.0 at index 1
		// Row 1: max is 6.0 at index 2
		// Row 2: max is 7.0 at index 1
		[]int64{1, 2, 1},
	}, -1)

	// Test ArgMax with select_last_index=true (selects last occurrence when there are ties)
	graphtest.RunTestGraphFn(t, "ArgMax-select-last-index", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{1.0, 4.0, 4.0, 2.0}, {3.0, 5.0, 5.0, 5.0}})

		node := &protos.NodeProto{
			OpType: "ArgMax",
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: 1},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
				{Name: "select_last_index", Type: protos.AttributeProto_INT, I: 1},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertArgMax(node, inputs),
		}
		return
	}, []any{
		// ArgMax along axis 1 with select_last_index: [2, 3]
		// Row 0: max is 4.0, last occurrence at index 2
		// Row 1: max is 5.0, last occurrence at index 3
		[]int64{2, 3},
	}, -1)

	// Test ArgMax with negative values
	graphtest.RunTestGraphFn(t, "ArgMax-negative-values", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{-5.0, -2.0, -8.0}, {-1.0, -9.0, -3.0}})

		node := &protos.NodeProto{
			OpType: "ArgMax",
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: 1},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertArgMax(node, inputs),
		}
		return
	}, []any{
		// ArgMax along axis 1: [1, 0]
		// Row 0: max is -2.0 at index 1
		// Row 1: max is -1.0 at index 0
		[]int64{1, 0},
	}, -1)
}

func TestArgMin(t *testing.T) {
	// Test ArgMin along axis 0 with keepdims=true
	graphtest.RunTestGraphFn(t, "ArgMin-axis0-keepdims", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{3.0, 2.0}, {1.0, 5.0}, {4.0, 1.0}})

		node := &protos.NodeProto{
			OpType: "ArgMin",
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: 0},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 1},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertArgMin(node, inputs),
		}
		return
	}, []any{
		// ArgMin along axis 0: [[1, 2]]
		// Column 0: min is 1.0 at index 1
		// Column 1: min is 1.0 at index 2
		[][]int64{{1, 2}},
	}, -1)

	// Test ArgMin along axis 1 without keepdims
	graphtest.RunTestGraphFn(t, "ArgMin-axis1-no-keepdims", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{5.0, 2.0, 8.0}, {1.0, 9.0, 3.0}, {7.0, 4.0, 6.0}})

		node := &protos.NodeProto{
			OpType: "ArgMin",
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: 1},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertArgMin(node, inputs),
		}
		return
	}, []any{
		// ArgMin along axis 1: [1, 0, 1]
		// Row 0: min is 2.0 at index 1
		// Row 1: min is 1.0 at index 0
		// Row 2: min is 4.0 at index 1
		[]int64{1, 0, 1},
	}, -1)

	// Test ArgMin with select_last_index=true
	graphtest.RunTestGraphFn(t, "ArgMin-select-last-index", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{2.0, 1.0, 1.0, 3.0}, {4.0, 2.0, 2.0, 2.0}})

		node := &protos.NodeProto{
			OpType: "ArgMin",
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: 1},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
				{Name: "select_last_index", Type: protos.AttributeProto_INT, I: 1},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertArgMin(node, inputs),
		}
		return
	}, []any{
		// ArgMin along axis 1 with select_last_index: [2, 3]
		// Row 0: min is 1.0, last occurrence at index 2
		// Row 1: min is 2.0, last occurrence at index 3
		[]int64{2, 3},
	}, -1)

	// Test ArgMin with 3D tensor
	graphtest.RunTestGraphFn(t, "ArgMin-3D-axis2", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][][]float32{
			{{5.0, 2.0, 8.0}, {1.0, 9.0, 3.0}},
			{{7.0, 4.0, 6.0}, {0.0, 5.0, 2.0}},
		})

		node := &protos.NodeProto{
			OpType: "ArgMin",
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: 2},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertArgMin(node, inputs),
		}
		return
	}, []any{
		// ArgMin along axis 2 (last axis): [[1, 0], [1, 0]]
		// [0,0]: min is 2.0 at index 1
		// [0,1]: min is 1.0 at index 0
		// [1,0]: min is 4.0 at index 1
		// [1,1]: min is 0.0 at index 0
		[][]int64{{1, 0}, {1, 0}},
	}, -1)
}

// Test ReduceOp with different dtypes to ensure type compatibility
func TestReduceOps_DifferentDTypes(t *testing.T) {
	// Test ReduceSum with int32
	graphtest.RunTestGraphFn(t, "ReduceSum-int32", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]int32{{1, 2, 3}, {4, 5, 6}})

		node := &protos.NodeProto{
			OpType: "ReduceSum",
			Attribute: []*protos.AttributeProto{
				{Name: "axes", Type: protos.AttributeProto_INTS, Ints: []int64{1}},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertReduceSum(nil, nil, node, inputs),
		}
		return
	}, []any{
		[]int32{6, 15},
	}, -1)

	// Test ReduceMax with float64
	graphtest.RunTestGraphFn(t, "ReduceMax-float64", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float64{{1.5, 2.5}, {3.5, 4.5}})

		node := &protos.NodeProto{
			OpType: "ReduceMax",
			Attribute: []*protos.AttributeProto{
				{Name: "axes", Type: protos.AttributeProto_INTS, Ints: []int64{0}},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertReduceMax(nil, nil, node, inputs),
		}
		return
	}, []any{
		[]float64{3.5, 4.5},
	}, -1)
}

// Test edge case: reduction on 1D tensor
func TestReduceOps_1D(t *testing.T) {
	graphtest.RunTestGraphFn(t, "ReduceSum-1D", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{1.0, 2.0, 3.0, 4.0, 5.0})

		node := &protos.NodeProto{
			OpType: "ReduceSum",
			Attribute: []*protos.AttributeProto{
				{Name: "axes", Type: protos.AttributeProto_INTS, Ints: []int64{0}},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertReduceSum(nil, nil, node, inputs),
		}
		return
	}, []any{
		float32(15.0),
	}, -1)

	graphtest.RunTestGraphFn(t, "ArgMax-1D", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{3.0, 1.0, 5.0, 2.0, 4.0})

		node := &protos.NodeProto{
			OpType: "ArgMax",
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: 0},
				{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0},
			},
		}

		inputs = []*Node{x}
		outputs = []*Node{
			convertArgMax(node, inputs),
		}
		return
	}, []any{
		// Max value 5.0 is at index 2
		int64(2),
	}, -1)
}
