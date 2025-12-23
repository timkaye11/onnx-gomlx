package onnx

import (
	"math"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/onnx-gomlx/internal/protos"
)

// TestSigmoid tests the Sigmoid activation function.
// Sigmoid(x) = 1 / (1 + exp(-x))
func TestSigmoid(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Sigmoid-basic", func(g *Graph) (inputs, outputs []*Node) {
		// Test with simple values: 0, positive, negative
		x := Const(g, []float32{-2.0, -1.0, 0.0, 1.0, 2.0})
		inputs = []*Node{x}
		outputs = []*Node{convertSigmoid([]*Node{x})}
		return
	}, []any{
		// sigmoid(-2) ≈ 0.1192, sigmoid(-1) ≈ 0.2689, sigmoid(0) = 0.5,
		// sigmoid(1) ≈ 0.7311, sigmoid(2) ≈ 0.8808
		[]float32{0.11920292, 0.26894142, 0.5, 0.7310586, 0.88079708},
	}, 1e-5)

	// Test with 2D tensor
	graphtest.RunTestGraphFn(t, "Sigmoid-2D", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{-1.0, 0.0}, {1.0, 2.0}})
		inputs = []*Node{x}
		outputs = []*Node{convertSigmoid([]*Node{x})}
		return
	}, []any{
		[][]float32{{0.26894142, 0.5}, {0.7310586, 0.88079708}},
	}, 1e-5)
}

// TestLeakyRelu tests the LeakyRelu activation function.
// LeakyRelu(x) = x if x >= 0, alpha * x otherwise
func TestLeakyRelu(t *testing.T) {
	// Test with default alpha (0.01)
	graphtest.RunTestGraphFn(t, "LeakyRelu-default-alpha", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{-2.0, -1.0, 0.0, 1.0, 2.0})
		node := &protos.NodeProto{
			OpType: "LeakyRelu",
			// Default alpha = 0.01
		}
		inputs = []*Node{x}
		outputs = []*Node{convertLeakyRelu(node, []*Node{x})}
		return
	}, []any{
		// For x < 0: alpha * x = 0.01 * x
		// -2.0 * 0.01 = -0.02, -1.0 * 0.01 = -0.01
		// For x >= 0: x
		[]float32{-0.02, -0.01, 0.0, 1.0, 2.0},
	}, 1e-6)

	// Test with custom alpha (0.2)
	graphtest.RunTestGraphFn(t, "LeakyRelu-custom-alpha", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{-10.0, -5.0, 0.0, 5.0, 10.0})
		node := &protos.NodeProto{
			OpType: "LeakyRelu",
			Attribute: []*protos.AttributeProto{
				{Name: "alpha", Type: protos.AttributeProto_FLOAT, F: 0.2},
			},
		}
		inputs = []*Node{x}
		outputs = []*Node{convertLeakyRelu(node, []*Node{x})}
		return
	}, []any{
		// For x < 0: 0.2 * x
		// -10 * 0.2 = -2, -5 * 0.2 = -1
		// For x >= 0: x
		[]float32{-2.0, -1.0, 0.0, 5.0, 10.0},
	}, 1e-6)

	// Test with 2D tensor
	graphtest.RunTestGraphFn(t, "LeakyRelu-2D", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{-4.0, -2.0}, {0.0, 2.0}, {4.0, 6.0}})
		node := &protos.NodeProto{
			OpType: "LeakyRelu",
			Attribute: []*protos.AttributeProto{
				{Name: "alpha", Type: protos.AttributeProto_FLOAT, F: 0.1},
			},
		}
		inputs = []*Node{x}
		outputs = []*Node{convertLeakyRelu(node, []*Node{x})}
		return
	}, []any{
		[][]float32{{-0.4, -0.2}, {0.0, 2.0}, {4.0, 6.0}},
	}, 1e-6)
}

// TestElu tests the Exponential Linear Unit activation function.
// Elu(x) = x if x >= 0, alpha * (exp(x) - 1) otherwise
func TestElu(t *testing.T) {
	// Test with default alpha (1.0)
	graphtest.RunTestGraphFn(t, "Elu-default-alpha", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{-2.0, -1.0, 0.0, 1.0, 2.0})
		node := &protos.NodeProto{
			OpType: "Elu",
			// Default alpha = 1.0
		}
		inputs = []*Node{x}
		outputs = []*Node{convertElu(node, []*Node{x})}
		return
	}, []any{
		// For x < 0: alpha * (exp(x) - 1)
		// alpha=1: exp(-2) - 1 ≈ -0.8647, exp(-1) - 1 ≈ -0.6321
		// For x >= 0: x
		[]float32{-0.86466473, -0.63212056, 0.0, 1.0, 2.0},
	}, 1e-5)

	// Test with custom alpha (0.5)
	graphtest.RunTestGraphFn(t, "Elu-custom-alpha", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{-2.0, -1.0, 0.0, 1.0, 2.0})
		node := &protos.NodeProto{
			OpType: "Elu",
			Attribute: []*protos.AttributeProto{
				{Name: "alpha", Type: protos.AttributeProto_FLOAT, F: 0.5},
			},
		}
		inputs = []*Node{x}
		outputs = []*Node{convertElu(node, []*Node{x})}
		return
	}, []any{
		// For x < 0: 0.5 * (exp(x) - 1)
		[]float32{-0.43233237, -0.31606028, 0.0, 1.0, 2.0},
	}, 1e-5)

	// Test with 2D tensor
	graphtest.RunTestGraphFn(t, "Elu-2D", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{-1.0, 0.0}, {1.0, 2.0}})
		node := &protos.NodeProto{
			OpType: "Elu",
			Attribute: []*protos.AttributeProto{
				{Name: "alpha", Type: protos.AttributeProto_FLOAT, F: 1.0},
			},
		}
		inputs = []*Node{x}
		outputs = []*Node{convertElu(node, []*Node{x})}
		return
	}, []any{
		[][]float32{{-0.63212056, 0.0}, {1.0, 2.0}},
	}, 1e-5)
}

// TestSoftplus tests the Softplus activation function.
// Softplus(x) = log(1 + exp(x))
func TestSoftplus(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Softplus-basic", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{-2.0, -1.0, 0.0, 1.0, 2.0})
		inputs = []*Node{x}
		outputs = []*Node{convertSoftplus([]*Node{x})}
		return
	}, []any{
		// softplus(-2) = log(1 + exp(-2)) ≈ 0.1269
		// softplus(-1) = log(1 + exp(-1)) ≈ 0.3133
		// softplus(0) = log(2) ≈ 0.6931
		// softplus(1) = log(1 + e) ≈ 1.3133
		// softplus(2) = log(1 + exp(2)) ≈ 2.1269
		[]float32{0.12692801, 0.31326168, 0.6931472, 1.3132616, 2.126928},
	}, 1e-5)

	// Test with 2D tensor
	graphtest.RunTestGraphFn(t, "Softplus-2D", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{-1.0, 0.0}, {1.0, 2.0}})
		inputs = []*Node{x}
		outputs = []*Node{convertSoftplus([]*Node{x})}
		return
	}, []any{
		[][]float32{{0.31326168, 0.6931472}, {1.3132616, 2.126928}},
	}, 1e-5)

	// Test with very small values (close to 0)
	graphtest.RunTestGraphFn(t, "Softplus-small-values", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{-5.0, 0.0, 5.0})
		inputs = []*Node{x}
		outputs = []*Node{convertSoftplus([]*Node{x})}
		return
	}, []any{
		// softplus(-5) ≈ 0.0067, softplus(0) = ln(2), softplus(5) ≈ 5.0067
		[]float32{0.006715348, 0.6931472, 5.0067153},
	}, 1e-5)
}

// TestSoftsign tests the Softsign activation function.
// Softsign(x) = x / (1 + |x|)
func TestSoftsign(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Softsign-basic", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{-2.0, -1.0, 0.0, 1.0, 2.0})
		inputs = []*Node{x}
		outputs = []*Node{convertSoftsign([]*Node{x})}
		return
	}, []any{
		// softsign(-2) = -2 / (1 + 2) = -2/3 ≈ -0.6667
		// softsign(-1) = -1 / (1 + 1) = -0.5
		// softsign(0) = 0 / 1 = 0
		// softsign(1) = 1 / 2 = 0.5
		// softsign(2) = 2 / 3 ≈ 0.6667
		[]float32{-0.6666667, -0.5, 0.0, 0.5, 0.6666667},
	}, 1e-6)

	// Test with 2D tensor
	graphtest.RunTestGraphFn(t, "Softsign-2D", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{-4.0, -2.0}, {0.0, 2.0}, {4.0, 8.0}})
		inputs = []*Node{x}
		outputs = []*Node{convertSoftsign([]*Node{x})}
		return
	}, []any{
		// softsign(-4) = -4/5 = -0.8
		// softsign(-2) = -2/3 ≈ -0.6667
		// softsign(0) = 0
		// softsign(2) = 2/3 ≈ 0.6667
		// softsign(4) = 4/5 = 0.8
		// softsign(8) = 8/9 ≈ 0.8889
		[][]float32{{-0.8, -0.6666667}, {0.0, 0.6666667}, {0.8, 0.8888889}},
	}, 1e-6)

	// Test with large values (should approach ±1)
	graphtest.RunTestGraphFn(t, "Softsign-large-values", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{-100.0, -10.0, 10.0, 100.0})
		inputs = []*Node{x}
		outputs = []*Node{convertSoftsign([]*Node{x})}
		return
	}, []any{
		// softsign(-100) = -100/101 ≈ -0.9901
		// softsign(-10) = -10/11 ≈ -0.9091
		// softsign(10) = 10/11 ≈ 0.9091
		// softsign(100) = 100/101 ≈ 0.9901
		[]float32{-0.990099, -0.90909094, 0.90909094, 0.990099},
	}, 1e-6)
}

// TestGelu tests the Gaussian Error Linear Unit activation function.
// Gelu has two approximation modes: "none" (exact) and "tanh" (approximate)
func TestGelu(t *testing.T) {
	// Test with exact mode (approximate="none")
	graphtest.RunTestGraphFn(t, "Gelu-exact", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{-2.0, -1.0, 0.0, 1.0, 2.0})
		node := &protos.NodeProto{
			OpType: "Gelu",
			Attribute: []*protos.AttributeProto{
				{Name: "approximate", Type: protos.AttributeProto_STRING, S: []byte("none")},
			},
		}
		inputs = []*Node{x}
		outputs = []*Node{convertGelu(node, []*Node{x})}
		return
	}, []any{
		// Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
		// gelu(-2) ≈ -0.04550
		// gelu(-1) ≈ -0.15880
		// gelu(0) = 0
		// gelu(1) ≈ 0.84134
		// gelu(2) ≈ 1.95505
		[]float32{-0.04550027, -0.15865526, 0.0, 0.84134477, 1.9545397},
	}, 1e-5)

	// Test with tanh approximation mode
	graphtest.RunTestGraphFn(t, "Gelu-tanh", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{-2.0, -1.0, 0.0, 1.0, 2.0})
		node := &protos.NodeProto{
			OpType: "Gelu",
			Attribute: []*protos.AttributeProto{
				{Name: "approximate", Type: protos.AttributeProto_STRING, S: []byte("tanh")},
			},
		}
		inputs = []*Node{x}
		outputs = []*Node{convertGelu(node, []*Node{x})}
		return
	}, []any{
		// Tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
		// Should be close to exact but slightly different
		[]float32{-0.04540223, -0.15880796, 0.0, 0.841192, 1.9545977},
	}, 1e-5)

	// Test with default mode (should be "none")
	graphtest.RunTestGraphFn(t, "Gelu-default", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{-1.0, 0.0, 1.0})
		node := &protos.NodeProto{
			OpType: "Gelu",
			// No approximate attribute - should default to "none"
		}
		inputs = []*Node{x}
		outputs = []*Node{convertGelu(node, []*Node{x})}
		return
	}, []any{
		[]float32{-0.15865526, 0.0, 0.84134477},
	}, 1e-5)

	// Test with 2D tensor
	graphtest.RunTestGraphFn(t, "Gelu-2D", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{-1.0, 0.0}, {1.0, 2.0}})
		node := &protos.NodeProto{
			OpType: "Gelu",
			Attribute: []*protos.AttributeProto{
				{Name: "approximate", Type: protos.AttributeProto_STRING, S: []byte("none")},
			},
		}
		inputs = []*Node{x}
		outputs = []*Node{convertGelu(node, []*Node{x})}
		return
	}, []any{
		[][]float32{{-0.15865526, 0.0}, {0.84134477, 1.9545397}},
	}, 1e-5)
}

// TestHardSigmoid tests the HardSigmoid activation function.
// HardSigmoid(x) = max(0, min(1, alpha * x + beta))
func TestHardSigmoid(t *testing.T) {
	// Test with default alpha and beta
	graphtest.RunTestGraphFn(t, "HardSigmoid-default", func(g *Graph) (inputs, outputs []*Node) {
		// Default: alpha=0.2, beta=0.5
		x := Const(g, []float32{-5.0, -2.5, 0.0, 2.5, 5.0})
		node := &protos.NodeProto{
			OpType: "HardSigmoid",
			// Default alpha=0.2, beta=0.5
		}
		inputs = []*Node{x}
		outputs = []*Node{convertHardSigmoid(node, []*Node{x})}
		return
	}, []any{
		// HardSigmoid(x) = max(0, min(1, 0.2*x + 0.5))
		// x=-5: 0.2*(-5) + 0.5 = -0.5, clamped to 0
		// x=-2.5: 0.2*(-2.5) + 0.5 = 0, stays 0
		// x=0: 0.2*0 + 0.5 = 0.5
		// x=2.5: 0.2*2.5 + 0.5 = 1.0
		// x=5: 0.2*5 + 0.5 = 1.5, clamped to 1
		[]float32{0.0, 0.0, 0.5, 1.0, 1.0},
	}, 1e-6)

	// Test with custom alpha and beta
	graphtest.RunTestGraphFn(t, "HardSigmoid-custom", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{-3.0, -1.0, 0.0, 1.0, 3.0})
		node := &protos.NodeProto{
			OpType: "HardSigmoid",
			Attribute: []*protos.AttributeProto{
				{Name: "alpha", Type: protos.AttributeProto_FLOAT, F: 0.5},
				{Name: "beta", Type: protos.AttributeProto_FLOAT, F: 0.25},
			},
		}
		inputs = []*Node{x}
		outputs = []*Node{convertHardSigmoid(node, []*Node{x})}
		return
	}, []any{
		// HardSigmoid(x) = max(0, min(1, 0.5*x + 0.25))
		// x=-3: 0.5*(-3) + 0.25 = -1.25, clamped to 0
		// x=-1: 0.5*(-1) + 0.25 = -0.25, clamped to 0
		// x=0: 0.5*0 + 0.25 = 0.25
		// x=1: 0.5*1 + 0.25 = 0.75
		// x=3: 0.5*3 + 0.25 = 1.75, clamped to 1
		[]float32{0.0, 0.0, 0.25, 0.75, 1.0},
	}, 1e-6)

	// Test with 2D tensor
	graphtest.RunTestGraphFn(t, "HardSigmoid-2D", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{-3.0, -1.0}, {0.0, 1.0}, {2.0, 3.0}})
		node := &protos.NodeProto{
			OpType: "HardSigmoid",
			Attribute: []*protos.AttributeProto{
				{Name: "alpha", Type: protos.AttributeProto_FLOAT, F: 0.2},
				{Name: "beta", Type: protos.AttributeProto_FLOAT, F: 0.5},
			},
		}
		inputs = []*Node{x}
		outputs = []*Node{convertHardSigmoid(node, []*Node{x})}
		return
	}, []any{
		// Using alpha=0.2, beta=0.5
		[][]float32{{0.0, 0.3}, {0.5, 0.7}, {0.9, 1.0}},
	}, 1e-6)
}

// TestLogSoftmax tests the LogSoftmax activation function.
// LogSoftmax(x) = log(softmax(x))
func TestLogSoftmax(t *testing.T) {
	// Test with 1D tensor (default axis=-1)
	graphtest.RunTestGraphFn(t, "LogSoftmax-1D", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, []float32{1.0, 2.0, 3.0})
		node := &protos.NodeProto{
			OpType: "LogSoftmax",
			// Default axis = -1
		}
		inputs = []*Node{x}
		outputs = []*Node{convertLogSoftmax(node, []*Node{x})}
		return
	}, []any{
		// First compute softmax:
		// exp_sum = exp(1) + exp(2) + exp(3) ≈ 2.7183 + 7.3891 + 20.0855 ≈ 30.1929
		// softmax = [exp(1)/sum, exp(2)/sum, exp(3)/sum]
		//         ≈ [0.09003, 0.24473, 0.66524]
		// logsoftmax = log(softmax)
		//            ≈ [-2.4076, -1.4076, -0.4076]
		[]float32{-2.4076061, -1.4076059, -0.40760595},
	}, 1e-5)

	// Test with 2D tensor, axis=-1 (last axis)
	graphtest.RunTestGraphFn(t, "LogSoftmax-2D-axis-1", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}})
		node := &protos.NodeProto{
			OpType: "LogSoftmax",
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: -1},
			},
		}
		inputs = []*Node{x}
		outputs = []*Node{convertLogSoftmax(node, []*Node{x})}
		return
	}, []any{
		// Each row normalized independently
		// Row 0: same as 1D case above
		// Row 1: [4,5,6] -> logsoftmax ≈ [-2.4076, -1.4076, -0.4076]
		[][]float32{{-2.4076061, -1.4076059, -0.40760595}, {-2.4076061, -1.4076059, -0.40760595}},
	}, 1e-5)

	// Test with 2D tensor, axis=0
	graphtest.RunTestGraphFn(t, "LogSoftmax-2D-axis-0", func(g *Graph) (inputs, outputs []*Node) {
		x := Const(g, [][]float32{{1.0, 2.0}, {3.0, 4.0}})
		node := &protos.NodeProto{
			OpType: "LogSoftmax",
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: 0},
			},
		}
		inputs = []*Node{x}
		outputs = []*Node{convertLogSoftmax(node, []*Node{x})}
		return
	}, []any{
		// Normalize over columns (axis 0)
		// Column 0: [1, 3] -> exp_sum = e^1 + e^3 ≈ 2.7183 + 20.0855 ≈ 22.8038
		//           softmax ≈ [0.1192, 0.8808], logsoftmax ≈ [-2.1269, -0.1269]
		// Column 1: [2, 4] -> exp_sum = e^2 + e^4 ≈ 7.3891 + 54.5982 ≈ 61.9873
		//           softmax ≈ [0.1192, 0.8808], logsoftmax ≈ [-2.1269, -0.1269]
		[][]float32{{-2.126928, -2.126928}, {-0.12692805, -0.12692805}},
	}, 1e-5)

	// Test with simple values for manual verification
	graphtest.RunTestGraphFn(t, "LogSoftmax-simple", func(g *Graph) (inputs, outputs []*Node) {
		// Using values where we can more easily verify
		x := Const(g, []float32{0.0, 0.0, 0.0})
		node := &protos.NodeProto{
			OpType: "LogSoftmax",
			Attribute: []*protos.AttributeProto{
				{Name: "axis", Type: protos.AttributeProto_INT, I: 0},
			},
		}
		inputs = []*Node{x}
		outputs = []*Node{convertLogSoftmax(node, []*Node{x})}
		return
	}, []any{
		// All equal values -> softmax = [1/3, 1/3, 1/3]
		// logsoftmax = log(1/3) ≈ -1.0986
		[]float32{float32(-math.Log(3)), float32(-math.Log(3)), float32(-math.Log(3))},
	}, 1e-5)
}
