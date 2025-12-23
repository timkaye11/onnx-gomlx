package onnx

import (
	"math"
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
)

func TestReciprocal(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Reciprocal-basic", func(g *Graph) (inputs, outputs []*Node) {
		// Test 1/x for simple values
		x := Const(g, []float32{1.0, 2.0, 4.0, 0.5})
		inputs = []*Node{x}
		outputs = []*Node{convertReciprocal([]*Node{x})}
		return
	}, []any{
		// Expected: 1/1=1, 1/2=0.5, 1/4=0.25, 1/0.5=2
		[]float32{1.0, 0.5, 0.25, 2.0},
	}, 1e-6)

	graphtest.RunTestGraphFn(t, "Reciprocal-2D", func(g *Graph) (inputs, outputs []*Node) {
		// Test with 2D tensor
		x := Const(g, [][]float32{{1.0, 2.0}, {4.0, 5.0}})
		inputs = []*Node{x}
		outputs = []*Node{convertReciprocal([]*Node{x})}
		return
	}, []any{
		[][]float32{{1.0, 0.5}, {0.25, 0.2}},
	}, 1e-6)

	graphtest.RunTestGraphFn(t, "Reciprocal-negative", func(g *Graph) (inputs, outputs []*Node) {
		// Test with negative values
		x := Const(g, []float32{-1.0, -2.0, -0.5})
		inputs = []*Node{x}
		outputs = []*Node{convertReciprocal([]*Node{x})}
		return
	}, []any{
		// Expected: 1/(-1)=-1, 1/(-2)=-0.5, 1/(-0.5)=-2
		[]float32{-1.0, -0.5, -2.0},
	}, 1e-6)
}

func TestRound(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Round-basic", func(g *Graph) (inputs, outputs []*Node) {
		// Test rounding - uses round-half-to-even (banker's rounding)
		x := Const(g, []float32{1.2, 1.5, 2.5, 3.7, -1.2, -1.5, -2.5})
		inputs = []*Node{x}
		outputs = []*Node{convertRound([]*Node{x})}
		return
	}, []any{
		// Round-half-to-even: 1.5→2 (even), 2.5→2 (even), -1.5→-2 (even), -2.5→-2 (even)
		[]float32{1.0, 2.0, 2.0, 4.0, -1.0, -2.0, -2.0},
	}, 1e-6)

	graphtest.RunTestGraphFn(t, "Round-2D", func(g *Graph) (inputs, outputs []*Node) {
		// Test with 2D tensor
		x := Const(g, [][]float32{{0.4, 0.6}, {1.5, 2.5}})
		inputs = []*Node{x}
		outputs = []*Node{convertRound([]*Node{x})}
		return
	}, []any{
		[][]float32{{0.0, 1.0}, {2.0, 2.0}},
	}, 1e-6)

	graphtest.RunTestGraphFn(t, "Round-already-integer", func(g *Graph) (inputs, outputs []*Node) {
		// Test values that are already integers
		x := Const(g, []float32{1.0, 2.0, 3.0, -5.0})
		inputs = []*Node{x}
		outputs = []*Node{convertRound([]*Node{x})}
		return
	}, []any{
		[]float32{1.0, 2.0, 3.0, -5.0},
	}, 1e-6)
}

func TestTan(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Tan-basic", func(g *Graph) (inputs, outputs []*Node) {
		// Test tangent at common angles (in radians)
		// tan(0)=0, tan(π/4)≈1, tan(-π/4)≈-1
		x := Const(g, []float32{0.0, float32(math.Pi / 4), float32(-math.Pi / 4)})
		inputs = []*Node{x}
		outputs = []*Node{convertTan([]*Node{x})}
		return
	}, []any{
		[]float32{0.0, 1.0, -1.0},
	}, 1e-6)

	graphtest.RunTestGraphFn(t, "Tan-2D", func(g *Graph) (inputs, outputs []*Node) {
		// Test with 2D tensor
		x := Const(g, [][]float32{
			{0.0, float32(math.Pi / 6)},          // tan(0)=0, tan(π/6)≈0.577
			{float32(math.Pi / 3), float32(0.5)}, // tan(π/3)≈1.732, tan(0.5)≈0.546
		})
		inputs = []*Node{x}
		outputs = []*Node{convertTan([]*Node{x})}
		return
	}, []any{
		[][]float32{
			{0.0, 0.57735},
			{1.73205, 0.54630},
		},
	}, 1e-4)
}

func TestSinh(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Sinh-basic", func(g *Graph) (inputs, outputs []*Node) {
		// Test hyperbolic sine
		// sinh(0)=0, sinh(1)≈1.175, sinh(-1)≈-1.175
		x := Const(g, []float32{0.0, 1.0, -1.0})
		inputs = []*Node{x}
		outputs = []*Node{convertSinh([]*Node{x})}
		return
	}, []any{
		[]float32{0.0, 1.17520, -1.17520},
	}, 1e-4)

	graphtest.RunTestGraphFn(t, "Sinh-2D", func(g *Graph) (inputs, outputs []*Node) {
		// Test with 2D tensor
		x := Const(g, [][]float32{{0.0, 0.5}, {1.0, 2.0}})
		inputs = []*Node{x}
		outputs = []*Node{convertSinh([]*Node{x})}
		return
	}, []any{
		// sinh(0)=0, sinh(0.5)≈0.521, sinh(1)≈1.175, sinh(2)≈3.627
		[][]float32{{0.0, 0.52110}, {1.17520, 3.62686}},
	}, 1e-4)
}

func TestCosh(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Cosh-basic", func(g *Graph) (inputs, outputs []*Node) {
		// Test hyperbolic cosine
		// cosh(0)=1, cosh(1)≈1.543, cosh(-1)≈1.543
		x := Const(g, []float32{0.0, 1.0, -1.0})
		inputs = []*Node{x}
		outputs = []*Node{convertCosh([]*Node{x})}
		return
	}, []any{
		[]float32{1.0, 1.54308, 1.54308},
	}, 1e-4)

	graphtest.RunTestGraphFn(t, "Cosh-2D", func(g *Graph) (inputs, outputs []*Node) {
		// Test with 2D tensor
		x := Const(g, [][]float32{{0.0, 0.5}, {1.0, 2.0}})
		inputs = []*Node{x}
		outputs = []*Node{convertCosh([]*Node{x})}
		return
	}, []any{
		// cosh(0)=1, cosh(0.5)≈1.128, cosh(1)≈1.543, cosh(2)≈3.762
		[][]float32{{1.0, 1.12763}, {1.54308, 3.76220}},
	}, 1e-4)
}

func TestAsin(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Asin-basic", func(g *Graph) (inputs, outputs []*Node) {
		// Test arcsine
		// asin(0)=0, asin(0.5)≈0.524 (π/6), asin(1)≈1.571 (π/2)
		x := Const(g, []float32{0.0, 0.5, -0.5, 1.0, -1.0})
		inputs = []*Node{x}
		outputs = []*Node{convertAsin([]*Node{x})}
		return
	}, []any{
		[]float32{0.0, float32(math.Pi / 6), float32(-math.Pi / 6), float32(math.Pi / 2), float32(-math.Pi / 2)},
	}, 1e-5)

	graphtest.RunTestGraphFn(t, "Asin-2D", func(g *Graph) (inputs, outputs []*Node) {
		// Test with 2D tensor
		x := Const(g, [][]float32{{0.0, 0.5}, {0.707, 1.0}})
		inputs = []*Node{x}
		outputs = []*Node{convertAsin([]*Node{x})}
		return
	}, []any{
		// asin(0)=0, asin(0.5)≈π/6, asin(0.707)≈π/4, asin(1)=π/2
		[][]float32{{0.0, 0.52360}, {0.77540, 1.57080}},
	}, 1e-4)
}

func TestAcos(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Acos-basic", func(g *Graph) (inputs, outputs []*Node) {
		// Test arccosine
		// acos(1)=0, acos(0.5)≈1.047 (π/3), acos(0)=π/2
		x := Const(g, []float32{1.0, 0.5, 0.0, -0.5, -1.0})
		inputs = []*Node{x}
		outputs = []*Node{convertAcos([]*Node{x})}
		return
	}, []any{
		[]float32{0.0, float32(math.Pi / 3), float32(math.Pi / 2), float32(2 * math.Pi / 3), float32(math.Pi)},
	}, 1e-5)

	graphtest.RunTestGraphFn(t, "Acos-2D", func(g *Graph) (inputs, outputs []*Node) {
		// Test with 2D tensor
		x := Const(g, [][]float32{{1.0, 0.707}, {0.5, 0.0}})
		inputs = []*Node{x}
		outputs = []*Node{convertAcos([]*Node{x})}
		return
	}, []any{
		// acos(1)=0, acos(0.707)≈π/4, acos(0.5)≈π/3, acos(0)=π/2
		[][]float32{{0.0, 0.79540}, {1.04720, 1.57080}},
	}, 1e-4)
}

func TestAtan(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Atan-basic", func(g *Graph) (inputs, outputs []*Node) {
		// Test arctangent
		// atan(0)=0, atan(1)≈0.785 (π/4), atan(-1)≈-π/4
		x := Const(g, []float32{0.0, 1.0, -1.0, 1000.0, -1000.0})
		inputs = []*Node{x}
		outputs = []*Node{convertAtan([]*Node{x})}
		return
	}, []any{
		// atan(large)→π/2, atan(-large)→-π/2
		[]float32{0.0, float32(math.Pi / 4), float32(-math.Pi / 4), float32(math.Pi / 2), float32(-math.Pi / 2)},
	}, 1e-5)

	graphtest.RunTestGraphFn(t, "Atan-2D", func(g *Graph) (inputs, outputs []*Node) {
		// Test with 2D tensor
		x := Const(g, [][]float32{{0.0, 0.577}, {1.0, 1.732}})
		inputs = []*Node{x}
		outputs = []*Node{convertAtan([]*Node{x})}
		return
	}, []any{
		// atan(0)=0, atan(0.577)≈π/6, atan(1)=π/4, atan(1.732)≈π/3
		[][]float32{{0.0, 0.52360}, {0.78540, 1.04720}},
	}, 1e-4)
}

func TestAsinh(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Asinh-basic", func(g *Graph) (inputs, outputs []*Node) {
		// Test inverse hyperbolic sine
		// asinh(0)=0, asinh(1)≈0.881, asinh(-1)≈-0.881
		x := Const(g, []float32{0.0, 1.0, -1.0, 2.0})
		inputs = []*Node{x}
		outputs = []*Node{convertAsinh([]*Node{x})}
		return
	}, []any{
		// asinh(0)=0, asinh(1)≈0.881, asinh(-1)≈-0.881, asinh(2)≈1.444
		[]float32{0.0, 0.88137, -0.88137, 1.44364},
	}, 1e-4)

	graphtest.RunTestGraphFn(t, "Asinh-2D", func(g *Graph) (inputs, outputs []*Node) {
		// Test with 2D tensor
		x := Const(g, [][]float32{{0.0, 0.5}, {1.0, 3.0}})
		inputs = []*Node{x}
		outputs = []*Node{convertAsinh([]*Node{x})}
		return
	}, []any{
		// asinh(0)=0, asinh(0.5)≈0.481, asinh(1)≈0.881, asinh(3)≈1.818
		[][]float32{{0.0, 0.48121}, {0.88137, 1.81845}},
	}, 1e-4)
}

func TestAcosh(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Acosh-basic", func(g *Graph) (inputs, outputs []*Node) {
		// Test inverse hyperbolic cosine (domain: x >= 1)
		// acosh(1)=0, acosh(2)≈1.317, acosh(3)≈1.763
		x := Const(g, []float32{1.0, 2.0, 3.0, 5.0})
		inputs = []*Node{x}
		outputs = []*Node{convertAcosh([]*Node{x})}
		return
	}, []any{
		// acosh(1)=0, acosh(2)≈1.317, acosh(3)≈1.763, acosh(5)≈2.292
		[]float32{0.0, 1.31696, 1.76275, 2.29243},
	}, 1e-4)

	graphtest.RunTestGraphFn(t, "Acosh-2D", func(g *Graph) (inputs, outputs []*Node) {
		// Test with 2D tensor
		x := Const(g, [][]float32{{1.0, 1.5}, {2.0, 4.0}})
		inputs = []*Node{x}
		outputs = []*Node{convertAcosh([]*Node{x})}
		return
	}, []any{
		// acosh(1)=0, acosh(1.5)≈0.962, acosh(2)≈1.317, acosh(4)≈2.063
		[][]float32{{0.0, 0.96242}, {1.31696, 2.06344}},
	}, 1e-4)
}

func TestAtanh(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Atanh-basic", func(g *Graph) (inputs, outputs []*Node) {
		// Test inverse hyperbolic tangent (domain: -1 < x < 1)
		// atanh(0)=0, atanh(0.5)≈0.549, atanh(-0.5)≈-0.549
		x := Const(g, []float32{0.0, 0.5, -0.5, 0.9})
		inputs = []*Node{x}
		outputs = []*Node{convertAtanh([]*Node{x})}
		return
	}, []any{
		// atanh(0)=0, atanh(0.5)≈0.549, atanh(-0.5)≈-0.549, atanh(0.9)≈1.472
		[]float32{0.0, 0.54931, -0.54931, 1.47222},
	}, 1e-4)

	graphtest.RunTestGraphFn(t, "Atanh-2D", func(g *Graph) (inputs, outputs []*Node) {
		// Test with 2D tensor
		x := Const(g, [][]float32{{0.0, 0.3}, {0.5, 0.7}})
		inputs = []*Node{x}
		outputs = []*Node{convertAtanh([]*Node{x})}
		return
	}, []any{
		// atanh(0)=0, atanh(0.3)≈0.310, atanh(0.5)≈0.549, atanh(0.7)≈0.867
		[][]float32{{0.0, 0.30952}, {0.54931, 0.86730}},
	}, 1e-4)
}

func TestSum(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Sum-two-tensors", func(g *Graph) (inputs, outputs []*Node) {
		// Test element-wise sum of two tensors
		x1 := Const(g, []float32{1.0, 2.0, 3.0})
		x2 := Const(g, []float32{4.0, 5.0, 6.0})
		inputs = []*Node{x1, x2}
		outputs = []*Node{convertSum([]*Node{x1, x2})}
		return
	}, []any{
		[]float32{5.0, 7.0, 9.0},
	}, 1e-6)

	graphtest.RunTestGraphFn(t, "Sum-three-tensors", func(g *Graph) (inputs, outputs []*Node) {
		// Test element-wise sum of three tensors
		x1 := Const(g, []float32{1.0, 2.0})
		x2 := Const(g, []float32{3.0, 4.0})
		x3 := Const(g, []float32{5.0, 6.0})
		inputs = []*Node{x1, x2, x3}
		outputs = []*Node{convertSum([]*Node{x1, x2, x3})}
		return
	}, []any{
		[]float32{9.0, 12.0},
	}, 1e-6)

	graphtest.RunTestGraphFn(t, "Sum-2D-tensors", func(g *Graph) (inputs, outputs []*Node) {
		// Test sum with 2D tensors
		x1 := Const(g, [][]float32{{1.0, 2.0}, {3.0, 4.0}})
		x2 := Const(g, [][]float32{{5.0, 6.0}, {7.0, 8.0}})
		inputs = []*Node{x1, x2}
		outputs = []*Node{convertSum([]*Node{x1, x2})}
		return
	}, []any{
		[][]float32{{6.0, 8.0}, {10.0, 12.0}},
	}, 1e-6)

	graphtest.RunTestGraphFn(t, "Sum-single-tensor", func(g *Graph) (inputs, outputs []*Node) {
		// Test sum with single tensor (should return same tensor)
		x := Const(g, []float32{1.0, 2.0, 3.0})
		inputs = []*Node{x}
		outputs = []*Node{convertSum([]*Node{x})}
		return
	}, []any{
		[]float32{1.0, 2.0, 3.0},
	}, 1e-6)

	graphtest.RunTestGraphFn(t, "Sum-broadcasting", func(g *Graph) (inputs, outputs []*Node) {
		// Test sum with broadcasting
		x1 := Const(g, [][]float32{{1.0, 2.0}, {3.0, 4.0}})
		x2 := Const(g, []float32{10.0, 20.0})
		inputs = []*Node{x1, x2}
		outputs = []*Node{convertSum([]*Node{x1, x2})}
		return
	}, []any{
		[][]float32{{11.0, 22.0}, {13.0, 24.0}},
	}, 1e-6)
}

func TestMean(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Mean-two-tensors", func(g *Graph) (inputs, outputs []*Node) {
		// Test element-wise mean of two tensors
		x1 := Const(g, []float32{1.0, 2.0, 3.0})
		x2 := Const(g, []float32{5.0, 6.0, 9.0})
		inputs = []*Node{x1, x2}
		outputs = []*Node{convertMean([]*Node{x1, x2})}
		return
	}, []any{
		// Mean: (1+5)/2=3, (2+6)/2=4, (3+9)/2=6
		[]float32{3.0, 4.0, 6.0},
	}, 1e-6)

	graphtest.RunTestGraphFn(t, "Mean-three-tensors", func(g *Graph) (inputs, outputs []*Node) {
		// Test element-wise mean of three tensors
		x1 := Const(g, []float32{1.0, 2.0})
		x2 := Const(g, []float32{3.0, 4.0})
		x3 := Const(g, []float32{5.0, 6.0})
		inputs = []*Node{x1, x2, x3}
		outputs = []*Node{convertMean([]*Node{x1, x2, x3})}
		return
	}, []any{
		// Mean: (1+3+5)/3=3, (2+4+6)/3=4
		[]float32{3.0, 4.0},
	}, 1e-6)

	graphtest.RunTestGraphFn(t, "Mean-2D-tensors", func(g *Graph) (inputs, outputs []*Node) {
		// Test mean with 2D tensors
		x1 := Const(g, [][]float32{{1.0, 2.0}, {3.0, 4.0}})
		x2 := Const(g, [][]float32{{5.0, 6.0}, {7.0, 8.0}})
		inputs = []*Node{x1, x2}
		outputs = []*Node{convertMean([]*Node{x1, x2})}
		return
	}, []any{
		[][]float32{{3.0, 4.0}, {5.0, 6.0}},
	}, 1e-6)

	graphtest.RunTestGraphFn(t, "Mean-single-tensor", func(g *Graph) (inputs, outputs []*Node) {
		// Test mean with single tensor (should return same tensor)
		x := Const(g, []float32{1.0, 2.0, 3.0})
		inputs = []*Node{x}
		outputs = []*Node{convertMean([]*Node{x})}
		return
	}, []any{
		[]float32{1.0, 2.0, 3.0},
	}, 1e-6)

	graphtest.RunTestGraphFn(t, "Mean-four-tensors", func(g *Graph) (inputs, outputs []*Node) {
		// Test mean with four tensors
		x1 := Const(g, []float32{1.0, 2.0})
		x2 := Const(g, []float32{3.0, 4.0})
		x3 := Const(g, []float32{5.0, 6.0})
		x4 := Const(g, []float32{7.0, 8.0})
		inputs = []*Node{x1, x2, x3, x4}
		outputs = []*Node{convertMean([]*Node{x1, x2, x3, x4})}
		return
	}, []any{
		// Mean: (1+3+5+7)/4=4, (2+4+6+8)/4=5
		[]float32{4.0, 5.0},
	}, 1e-6)

	graphtest.RunTestGraphFn(t, "Mean-broadcasting", func(g *Graph) (inputs, outputs []*Node) {
		// Test mean with broadcasting
		x1 := Const(g, [][]float32{{2.0, 4.0}, {6.0, 8.0}})
		x2 := Const(g, []float32{10.0, 20.0})
		inputs = []*Node{x1, x2}
		outputs = []*Node{convertMean([]*Node{x1, x2})}
		return
	}, []any{
		// Mean: (2+10)/2=6, (4+20)/2=12, (6+10)/2=8, (8+20)/2=14
		[][]float32{{6.0, 12.0}, {8.0, 14.0}},
	}, 1e-6)
}
