package onnx

// This file implements ONNX mathematical operators.

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/onnx-gomlx/internal/protos"
)

// convertReciprocal converts a ONNX Reciprocal node to a GoMLX node.
// Reciprocal(x) = 1/x
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Reciprocal.html
func convertReciprocal(inputs []*Node) *Node {
	x := inputs[0]
	one := ScalarOne(x.Graph(), x.DType())
	return Div(one, x)
}

// convertRound converts a ONNX Round node to a GoMLX node.
// Round(x) rounds to the nearest integer, with halfway cases rounded to the nearest even number.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Round.html
func convertRound(inputs []*Node) *Node {
	return Round(inputs[0])
}

// convertMod converts a ONNX Mod node to a GoMLX node.
// Mod computes the remainder of division.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Mod.html
func convertMod(node *protos.NodeProto, inputs []*Node) *Node {
	fmod := getBoolAttrOr(node, "fmod", false)
	operands := onnxImplicitExpansion(inputs)
	a, b := operands[0], operands[1]

	if fmod || a.DType().IsFloat() {
		// fmod semantics: result has same sign as dividend (a)
		// fmod(a, b) = a - trunc(a/b) * b
		quotient := Div(a, b)
		// Truncate towards zero
		truncQuotient := Where(GreaterOrEqual(quotient, ScalarZero(a.Graph(), quotient.DType())),
			Floor(quotient), Ceil(quotient))
		return Sub(a, Mul(truncQuotient, b))
	}

	// Integer modulo semantics: result has same sign as divisor (b)
	// This matches Python's % operator behavior
	return Mod(a, b)
}

// convertTan converts a ONNX Tan node to a GoMLX node.
// Tan(x) = sin(x) / cos(x)
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Tan.html
func convertTan(inputs []*Node) *Node {
	x := inputs[0]
	return Div(Sin(x), Cos(x))
}

// convertSinh converts a ONNX Sinh node to a GoMLX node.
// Sinh(x) = (exp(x) - exp(-x)) / 2
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Sinh.html
func convertSinh(inputs []*Node) *Node {
	x := inputs[0]
	two := Scalar(x.Graph(), x.DType(), 2.0)
	return Div(Sub(Exp(x), Exp(Neg(x))), two)
}

// convertCosh converts a ONNX Cosh node to a GoMLX node.
// Cosh(x) = (exp(x) + exp(-x)) / 2
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Cosh.html
func convertCosh(inputs []*Node) *Node {
	x := inputs[0]
	two := Scalar(x.Graph(), x.DType(), 2.0)
	return Div(Add(Exp(x), Exp(Neg(x))), two)
}

// convertAsin converts a ONNX Asin node to a GoMLX node.
// Asin computes the arcsine (inverse sine) of the input.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Asin.html
func convertAsin(inputs []*Node) *Node {
	return Asin(inputs[0])
}

// convertAcos converts a ONNX Acos node to a GoMLX node.
// Acos computes the arccosine (inverse cosine) of the input.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Acos.html
func convertAcos(inputs []*Node) *Node {
	return Acos(inputs[0])
}

// convertAtan converts a ONNX Atan node to a GoMLX node.
// Atan computes the arctangent (inverse tangent) of the input.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Atan.html
func convertAtan(inputs []*Node) *Node {
	return Atan(inputs[0])
}

// convertAsinh converts a ONNX Asinh node to a GoMLX node.
// Asinh(x) = ln(x + sqrt(x^2 + 1))
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Asinh.html
func convertAsinh(inputs []*Node) *Node {
	x := inputs[0]
	one := ScalarOne(x.Graph(), x.DType())
	// Asinh(x) = ln(x + sqrt(x^2 + 1))
	return Log(Add(x, Sqrt(Add(Mul(x, x), one))))
}

// convertAcosh converts a ONNX Acosh node to a GoMLX node.
// Acosh(x) = ln(x + sqrt(x^2 - 1))
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Acosh.html
func convertAcosh(inputs []*Node) *Node {
	x := inputs[0]
	one := ScalarOne(x.Graph(), x.DType())
	// Acosh(x) = ln(x + sqrt(x^2 - 1))
	return Log(Add(x, Sqrt(Sub(Mul(x, x), one))))
}

// convertAtanh converts a ONNX Atanh node to a GoMLX node.
// Atanh(x) = 0.5 * ln((1 + x) / (1 - x))
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Atanh.html
func convertAtanh(inputs []*Node) *Node {
	x := inputs[0]
	one := ScalarOne(x.Graph(), x.DType())
	half := Scalar(x.Graph(), x.DType(), 0.5)
	// Atanh(x) = 0.5 * ln((1 + x) / (1 - x))
	return Mul(half, Log(Div(Add(one, x), Sub(one, x))))
}

// convertIsNaN converts a ONNX IsNaN node to a GoMLX node.
// IsNaN returns true for each element that is NaN.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__IsNaN.html
func convertIsNaN(inputs []*Node) *Node {
	return IsNaN(inputs[0])
}

// convertIsInf converts a ONNX IsInf node to a GoMLX node.
// IsInf returns true for each element that is positive or negative infinity.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__IsInf.html
func convertIsInf(node *protos.NodeProto, inputs []*Node) *Node {
	detectPositive := getBoolAttrOr(node, "detect_positive", true)
	detectNegative := getBoolAttrOr(node, "detect_negative", true)
	x := inputs[0]

	posInf := IsInf(x, 1)
	negInf := IsInf(x, -1)

	if detectPositive && detectNegative {
		return LogicalOr(posInf, negInf)
	} else if detectPositive {
		return posInf
	} else if detectNegative {
		return negInf
	}
	// If both are false, return all false
	return Zeros(x.Graph(), shapes.Make(dtypes.Bool, x.Shape().Dimensions...))
}

// convertBitShift converts a ONNX BitShift node to a GoMLX node.
// BitShift performs bitwise left or right shift.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__BitShift.html
func convertBitShift(node *protos.NodeProto, inputs []*Node) *Node {
	direction := getStringAttrOr(node, "direction", "LEFT")
	operands := onnxImplicitExpansion(inputs)
	x, y := operands[0], operands[1]

	if direction == "LEFT" {
		return ShiftLeft(x, y)
	}
	// RIGHT shift - this is logical (unsigned) shift
	return ShiftRightLogical(x, y)
}

// convertSum converts a ONNX Sum node to a GoMLX node.
// Sum computes the element-wise sum of all input tensors.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Sum.html
func convertSum(inputs []*Node) *Node {
	if len(inputs) == 0 {
		exceptions.Panicf("Sum requires at least one input")
	}
	if len(inputs) == 1 {
		return inputs[0]
	}

	// Broadcast all inputs to common shape and sum them
	inputs = onnxBroadcastToCommonShape(inputs)
	result := inputs[0]
	for i := 1; i < len(inputs); i++ {
		result = Add(result, inputs[i])
	}
	return result
}

// convertMean converts a ONNX Mean node to a GoMLX node.
// Mean computes the element-wise mean of all input tensors.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Mean.html
func convertMean(inputs []*Node) *Node {
	if len(inputs) == 0 {
		exceptions.Panicf("Mean requires at least one input")
	}
	if len(inputs) == 1 {
		return inputs[0]
	}

	// Broadcast all inputs to common shape and compute mean
	inputs = onnxBroadcastToCommonShape(inputs)
	result := inputs[0]
	for i := 1; i < len(inputs); i++ {
		result = Add(result, inputs[i])
	}
	n := Scalar(result.Graph(), result.DType(), float64(len(inputs)))
	return Div(result, n)
}
