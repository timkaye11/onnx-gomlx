package onnx

// This file implements ONNX activation function operators.

import (
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/onnx-gomlx/internal/protos"
)

// convertSigmoid converts a ONNX Sigmoid node to a GoMLX node.
// Sigmoid(x) = 1 / (1 + exp(-x))
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Sigmoid.html
func convertSigmoid(inputs []*Node) *Node {
	return Sigmoid(inputs[0])
}

// convertLeakyRelu converts a ONNX LeakyRelu node to a GoMLX node.
// LeakyRelu(x) = x if x >= 0, alpha * x otherwise
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__LeakyRelu.html
func convertLeakyRelu(node *protos.NodeProto, inputs []*Node) *Node {
	alpha := getFloatAttrOr(node, "alpha", 0.01)
	x := inputs[0]
	alphaNode := Scalar(x.Graph(), x.DType(), alpha)
	// LeakyRelu(x) = max(x, alpha * x) when alpha < 1
	// Or equivalently: x if x >= 0, alpha * x otherwise
	zero := ScalarZero(x.Graph(), x.DType())
	return Where(GreaterOrEqual(x, zero), x, Mul(alphaNode, x))
}

// convertPRelu converts a ONNX PRelu node to a GoMLX node.
// PRelu(x, slope) = x if x >= 0, slope * x otherwise
// Unlike LeakyRelu, slope is a learned parameter tensor.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__PRelu.html
func convertPRelu(inputs []*Node) *Node {
	x := inputs[0]
	slope := inputs[1]
	// Broadcast slope to x's shape if needed
	operands := onnxBroadcastToCommonShape([]*Node{x, slope})
	x, slope = operands[0], operands[1]
	zero := ScalarZero(x.Graph(), x.DType())
	return Where(GreaterOrEqual(x, zero), x, Mul(slope, x))
}

// convertElu converts a ONNX Elu node to a GoMLX node.
// Elu(x) = x if x >= 0, alpha * (exp(x) - 1) otherwise
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Elu.html
func convertElu(node *protos.NodeProto, inputs []*Node) *Node {
	alpha := getFloatAttrOr(node, "alpha", 1.0)
	x := inputs[0]
	alphaNode := Scalar(x.Graph(), x.DType(), alpha)
	zero := ScalarZero(x.Graph(), x.DType())
	one := ScalarOne(x.Graph(), x.DType())
	// Elu(x) = x if x >= 0, alpha * (exp(x) - 1) otherwise
	return Where(GreaterOrEqual(x, zero), x, Mul(alphaNode, Sub(Exp(x), one)))
}

// convertSelu converts a ONNX Selu node to a GoMLX node.
// Selu(x) = gamma * (x if x >= 0, alpha * (exp(x) - 1) otherwise)
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Selu.html
func convertSelu(node *protos.NodeProto, inputs []*Node) *Node {
	// Default values from the SELU paper
	alpha := getFloatAttrOr(node, "alpha", 1.6732632423543772)
	gamma := getFloatAttrOr(node, "gamma", 1.0507009873554805)
	x := inputs[0]
	alphaNode := Scalar(x.Graph(), x.DType(), alpha)
	gammaNode := Scalar(x.Graph(), x.DType(), gamma)
	zero := ScalarZero(x.Graph(), x.DType())
	one := ScalarOne(x.Graph(), x.DType())
	// Selu(x) = gamma * (x if x >= 0, alpha * (exp(x) - 1) otherwise)
	inner := Where(GreaterOrEqual(x, zero), x, Mul(alphaNode, Sub(Exp(x), one)))
	return Mul(gammaNode, inner)
}

// convertCelu converts a ONNX Celu node to a GoMLX node.
// Celu(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Celu.html
func convertCelu(node *protos.NodeProto, inputs []*Node) *Node {
	alpha := getFloatAttrOr(node, "alpha", 1.0)
	x := inputs[0]
	alphaNode := Scalar(x.Graph(), x.DType(), alpha)
	zero := ScalarZero(x.Graph(), x.DType())
	one := ScalarOne(x.Graph(), x.DType())
	// Celu(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
	positivePart := Max(zero, x)
	negativePart := Min(zero, Mul(alphaNode, Sub(Exp(Div(x, alphaNode)), one)))
	return Add(positivePart, negativePart)
}

// convertSoftplus converts a ONNX Softplus node to a GoMLX node.
// Softplus(x) = log(1 + exp(x))
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Softplus.html
func convertSoftplus(inputs []*Node) *Node {
	x := inputs[0]
	one := ScalarOne(x.Graph(), x.DType())
	// Softplus(x) = log(1 + exp(x))
	// For numerical stability, use: softplus(x) = max(x, 0) + log(1 + exp(-|x|))
	return Log(Add(one, Exp(x)))
}

// convertSoftsign converts a ONNX Softsign node to a GoMLX node.
// Softsign(x) = x / (1 + |x|)
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Softsign.html
func convertSoftsign(inputs []*Node) *Node {
	x := inputs[0]
	one := ScalarOne(x.Graph(), x.DType())
	// Softsign(x) = x / (1 + |x|)
	return Div(x, Add(one, Abs(x)))
}

// convertHardSigmoid converts a ONNX HardSigmoid node to a GoMLX node.
// HardSigmoid(x) = max(0, min(1, alpha * x + beta))
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__HardSigmoid.html
func convertHardSigmoid(node *protos.NodeProto, inputs []*Node) *Node {
	alpha := getFloatAttrOr(node, "alpha", 0.2)
	beta := getFloatAttrOr(node, "beta", 0.5)
	x := inputs[0]
	alphaNode := Scalar(x.Graph(), x.DType(), alpha)
	betaNode := Scalar(x.Graph(), x.DType(), beta)
	zero := ScalarZero(x.Graph(), x.DType())
	one := ScalarOne(x.Graph(), x.DType())
	// HardSigmoid(x) = max(0, min(1, alpha * x + beta))
	linear := Add(Mul(alphaNode, x), betaNode)
	return Max(zero, Min(one, linear))
}

// convertHardSwish converts a ONNX HardSwish node to a GoMLX node.
// HardSwish(x) = x * max(0, min(1, (x + 3) / 6))
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__HardSwish.html
func convertHardSwish(inputs []*Node) *Node {
	x := inputs[0]
	three := Scalar(x.Graph(), x.DType(), 3.0)
	six := Scalar(x.Graph(), x.DType(), 6.0)
	zero := ScalarZero(x.Graph(), x.DType())
	one := ScalarOne(x.Graph(), x.DType())
	// HardSwish(x) = x * max(0, min(1, (x + 3) / 6))
	inner := Div(Add(x, three), six)
	return Mul(x, Max(zero, Min(one, inner)))
}

// convertThresholdedRelu converts a ONNX ThresholdedRelu node to a GoMLX node.
// ThresholdedRelu(x) = x if x > alpha, 0 otherwise
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__ThresholdedRelu.html
func convertThresholdedRelu(node *protos.NodeProto, inputs []*Node) *Node {
	alpha := getFloatAttrOr(node, "alpha", 1.0)
	x := inputs[0]
	alphaNode := Scalar(x.Graph(), x.DType(), alpha)
	zero := ScalarZero(x.Graph(), x.DType())
	// ThresholdedRelu(x) = x if x > alpha, 0 otherwise
	return Where(GreaterThan(x, alphaNode), x, zero)
}

// convertGelu converts a ONNX Gelu node to a GoMLX node.
// Gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2))) for approximate="none"
// Gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) for approximate="tanh"
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Gelu.html
func convertGelu(node *protos.NodeProto, inputs []*Node) *Node {
	approximate := getStringAttrOr(node, "approximate", "none")
	x := inputs[0]
	half := Scalar(x.Graph(), x.DType(), 0.5)
	one := ScalarOne(x.Graph(), x.DType())

	if approximate == "tanh" {
		// Gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
		sqrt2OverPi := Scalar(x.Graph(), x.DType(), 0.7978845608028654) // sqrt(2/pi)
		coeff := Scalar(x.Graph(), x.DType(), 0.044715)
		xCubed := Mul(x, Mul(x, x))
		inner := Mul(sqrt2OverPi, Add(x, Mul(coeff, xCubed)))
		return Mul(half, Mul(x, Add(one, Tanh(inner))))
	}

	// Exact Gelu: 0.5 * x * (1 + erf(x / sqrt(2)))
	sqrtTwo := Scalar(x.Graph(), x.DType(), 1.4142135623730951)
	return Mul(half, Mul(x, Add(one, Erf(Div(x, sqrtTwo)))))
}

// convertMish converts a ONNX Mish node to a GoMLX node.
// Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Mish.html
func convertMish(inputs []*Node) *Node {
	x := inputs[0]
	one := ScalarOne(x.Graph(), x.DType())
	// Mish(x) = x * tanh(ln(1 + exp(x)))
	softplus := Log(Add(one, Exp(x)))
	return Mul(x, Tanh(softplus))
}

// convertLogSoftmax converts a ONNX LogSoftmax node to a GoMLX node.
// LogSoftmax(x) = log(softmax(x))
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__LogSoftmax.html
func convertLogSoftmax(node *protos.NodeProto, inputs []*Node) *Node {
	axis := getIntAttrOr(node, "axis", -1)
	x := inputs[0]
	// LogSoftmax = x - log(sum(exp(x)))
	// For numerical stability: x - max(x) - log(sum(exp(x - max(x))))
	return LogSoftmax(x, axis)
}

// convertHardmax converts a ONNX Hardmax node to a GoMLX node.
// Hardmax(x) returns a tensor with 1 at the position of max along axis, 0 elsewhere.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Hardmax.html
func convertHardmax(node *protos.NodeProto, inputs []*Node) *Node {
	axis := getIntAttrOr(node, "axis", -1)
	x := inputs[0]
	axis = AdjustAxisToOperandRank(x, axis)

	// Get indices of max values along axis
	maxIndices := ArgMax(x, axis)
	// Expand to match original shape
	maxIndices = ExpandAxes(maxIndices, axis)

	// Create one-hot encoding
	// Compare each position's index with the max index
	g := x.Graph()
	iotaShape := x.Shape().Clone()
	indices := Iota(g, iotaShape, axis)
	// Convert maxIndices to same dtype as indices for comparison
	maxIndices = ConvertDType(maxIndices, indices.DType())
	// Broadcast maxIndices to match x shape
	maxIndices = BroadcastToDims(maxIndices, x.Shape().Dimensions...)

	// Create mask where index equals max index
	mask := Equal(indices, maxIndices)
	one := ScalarOne(g, x.DType())
	zero := ScalarZero(g, x.DType())
	return Where(mask, one, zero)
}

// convertShrink converts a ONNX Shrink node to a GoMLX node.
// Shrink implements the shrinkage function used in some regularization techniques.
// If x < -lambd, y = x + bias
// If x > lambd, y = x - bias
// Otherwise, y = 0
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Shrink.html
func convertShrink(node *protos.NodeProto, inputs []*Node) *Node {
	bias := getFloatAttrOr(node, "bias", 0.0)
	lambd := getFloatAttrOr(node, "lambd", 0.5)
	x := inputs[0]

	biasNode := Scalar(x.Graph(), x.DType(), bias)
	lambdNode := Scalar(x.Graph(), x.DType(), lambd)
	negLambdNode := Neg(lambdNode)
	zero := ScalarZero(x.Graph(), x.DType())

	// If x < -lambd, y = x + bias
	// If x > lambd, y = x - bias
	// Otherwise, y = 0
	lessThanNegLambd := LessThan(x, negLambdNode)
	greaterThanLambd := GreaterThan(x, lambdNode)

	result := Where(lessThanNegLambd, Add(x, biasNode), zero)
	result = Where(greaterThanLambd, Sub(x, biasNode), result)
	return result
}
