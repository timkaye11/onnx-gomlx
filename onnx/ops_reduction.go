package onnx

// This file implements ONNX reduction operators.

import (
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/onnx-gomlx/internal/protos"
	"github.com/pkg/errors"
)

// convertReduceMax converts a ONNX ReduceMax node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__ReduceMax.html
func convertReduceMax(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	return convertReduceOp(m, convertedOutputs, node, inputs, ReduceMax, ReduceAllMax, "ReduceMax")
}

// convertReduceMin converts a ONNX ReduceMin node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__ReduceMin.html
func convertReduceMin(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	return convertReduceOp(m, convertedOutputs, node, inputs, ReduceMin, ReduceAllMin, "ReduceMin")
}

// convertReduceSum converts a ONNX ReduceSum node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__ReduceSum.html
func convertReduceSum(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	return convertReduceOp(m, convertedOutputs, node, inputs, ReduceSum, ReduceAllSum, "ReduceSum")
}

// convertReduceProd converts a ONNX ReduceProd node to a GoMLX node.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__ReduceProd.html
func convertReduceProd(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	return convertReduceOp(m, convertedOutputs, node, inputs, ReduceProd, ReduceAllProd, "ReduceProd")
}

// reduceAxisFn is a function type for reduction operations along specific axes.
type reduceAxisFn func(operand *Node, reduceAxes ...int) *Node

// reduceAllFn is a function type for reduction operations over all axes.
type reduceAllFn func(operand *Node) *Node

// convertReduceOp is a generic converter for reduction operations.
func convertReduceOp(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node,
	reduceAxis reduceAxisFn, reduceAll reduceAllFn, opName string) *Node {
	operand := inputs[0]
	keepDims := getIntAttrOr(node, "keepdims", 1) > 0
	noOpIfEmpty := getIntAttrOr(node, "noop_with_empty_axes", 0) > 0

	var axes []int
	if len(inputs) > 1 && inputs[1] != nil {
		if !inputs[1].DType().IsInt() {
			exceptions.Panicf("axes must be integer, got %s for node %s", inputs[1].DType(), nodeToString(node))
		}

		axesT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
		if err != nil {
			panic(errors.WithMessagef(err, "while converting 'axes' for node %s", nodeToString(node)))
		}
		axes = tensorToInts(axesT)
	}

	axesFromAttr := getIntsAttrOr(node, "axes", nil)
	if len(axesFromAttr) > 0 {
		if len(axes) > 0 {
			exceptions.Panicf("%s(operand, [axes]): axes input and axes attribute cannot be used together for node %s", opName, nodeToString(node))
		}
		axes = axesFromAttr
	}

	// If there are no axes to reduce, behavior depends on noop_with_empty_axes
	if len(axes) == 0 {
		if noOpIfEmpty {
			return Identity(operand)
		} else {
			res := reduceAll(operand)
			if keepDims {
				res = ExpandLeftToRank(res, operand.Rank())
			}
			return res
		}
	}

	if !keepDims {
		return reduceAxis(operand, axes...)
	} else {
		return ReduceAndKeep(operand, reduceAxis, axes...)
	}
}

// convertReduceL1 converts a ONNX ReduceL1 node to a GoMLX node.
// ReduceL1 computes the L1 norm (sum of absolute values) along specified axes.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__ReduceL1.html
func convertReduceL1(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	operand := inputs[0]
	keepDims := getIntAttrOr(node, "keepdims", 1) > 0
	noOpIfEmpty := getIntAttrOr(node, "noop_with_empty_axes", 0) > 0

	axes := getReduceAxes(m, convertedOutputs, node, inputs, "ReduceL1")

	// Take absolute value first
	absOperand := Abs(operand)

	if len(axes) == 0 {
		if noOpIfEmpty {
			return Identity(operand)
		} else {
			res := ReduceAllSum(absOperand)
			if keepDims {
				res = ExpandLeftToRank(res, operand.Rank())
			}
			return res
		}
	}

	if !keepDims {
		return ReduceSum(absOperand, axes...)
	} else {
		return ReduceAndKeep(absOperand, ReduceSum, axes...)
	}
}

// convertReduceL2 converts a ONNX ReduceL2 node to a GoMLX node.
// ReduceL2 computes the L2 norm (Euclidean norm) along specified axes.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__ReduceL2.html
func convertReduceL2(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	operand := inputs[0]
	keepDims := getIntAttrOr(node, "keepdims", 1) > 0
	noOpIfEmpty := getIntAttrOr(node, "noop_with_empty_axes", 0) > 0

	axes := getReduceAxes(m, convertedOutputs, node, inputs, "ReduceL2")

	// Square the values first
	squaredOperand := Mul(operand, operand)

	if len(axes) == 0 {
		if noOpIfEmpty {
			return Identity(operand)
		} else {
			res := Sqrt(ReduceAllSum(squaredOperand))
			if keepDims {
				res = ExpandLeftToRank(res, operand.Rank())
			}
			return res
		}
	}

	var res *Node
	if !keepDims {
		res = Sqrt(ReduceSum(squaredOperand, axes...))
	} else {
		res = Sqrt(ReduceAndKeep(squaredOperand, ReduceSum, axes...))
	}
	return res
}

// convertReduceLogSum converts a ONNX ReduceLogSum node to a GoMLX node.
// ReduceLogSum computes log(sum(x)) along specified axes.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__ReduceLogSum.html
func convertReduceLogSum(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	operand := inputs[0]
	keepDims := getIntAttrOr(node, "keepdims", 1) > 0
	noOpIfEmpty := getIntAttrOr(node, "noop_with_empty_axes", 0) > 0

	axes := getReduceAxes(m, convertedOutputs, node, inputs, "ReduceLogSum")

	if len(axes) == 0 {
		if noOpIfEmpty {
			return Identity(operand)
		} else {
			res := Log(ReduceAllSum(operand))
			if keepDims {
				res = ExpandLeftToRank(res, operand.Rank())
			}
			return res
		}
	}

	var res *Node
	if !keepDims {
		res = Log(ReduceSum(operand, axes...))
	} else {
		res = Log(ReduceAndKeep(operand, ReduceSum, axes...))
	}
	return res
}

// convertReduceLogSumExp converts a ONNX ReduceLogSumExp node to a GoMLX node.
// ReduceLogSumExp computes log(sum(exp(x))) along specified axes.
// This is numerically stable implementation using the log-sum-exp trick.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__ReduceLogSumExp.html
func convertReduceLogSumExp(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	operand := inputs[0]
	keepDims := getIntAttrOr(node, "keepdims", 1) > 0
	noOpIfEmpty := getIntAttrOr(node, "noop_with_empty_axes", 0) > 0

	axes := getReduceAxes(m, convertedOutputs, node, inputs, "ReduceLogSumExp")

	if len(axes) == 0 {
		if noOpIfEmpty {
			return Identity(operand)
		} else {
			// For numerical stability: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
			maxVal := ReduceAllMax(operand)
			shifted := Sub(operand, maxVal)
			res := Add(maxVal, Log(ReduceAllSum(Exp(shifted))))
			if keepDims {
				res = ExpandLeftToRank(res, operand.Rank())
			}
			return res
		}
	}

	// For numerical stability: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
	var maxVal *Node
	if keepDims {
		maxVal = ReduceAndKeep(operand, ReduceMax, axes...)
	} else {
		maxVal = ReduceMax(operand, axes...)
		// Need to expand maxVal back for subtraction
		maxVal = ExpandLeftToRank(maxVal, operand.Rank())
	}

	shifted := Sub(operand, maxVal)

	var res *Node
	if !keepDims {
		// Re-compute maxVal without keepdims for the final addition
		maxValNoKeep := ReduceMax(operand, axes...)
		res = Add(maxValNoKeep, Log(ReduceSum(Exp(shifted), axes...)))
	} else {
		res = Add(maxVal, Log(ReduceAndKeep(Exp(shifted), ReduceSum, axes...)))
	}
	return res
}

// convertReduceSumSquare converts a ONNX ReduceSumSquare node to a GoMLX node.
// ReduceSumSquare computes sum(x^2) along specified axes.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__ReduceSumSquare.html
func convertReduceSumSquare(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	operand := inputs[0]
	keepDims := getIntAttrOr(node, "keepdims", 1) > 0
	noOpIfEmpty := getIntAttrOr(node, "noop_with_empty_axes", 0) > 0

	axes := getReduceAxes(m, convertedOutputs, node, inputs, "ReduceSumSquare")

	// Square the values first
	squaredOperand := Mul(operand, operand)

	if len(axes) == 0 {
		if noOpIfEmpty {
			return Identity(operand)
		} else {
			res := ReduceAllSum(squaredOperand)
			if keepDims {
				res = ExpandLeftToRank(res, operand.Rank())
			}
			return res
		}
	}

	if !keepDims {
		return ReduceSum(squaredOperand, axes...)
	} else {
		return ReduceAndKeep(squaredOperand, ReduceSum, axes...)
	}
}

// getReduceAxes is a helper function to extract axes from either input or attribute.
func getReduceAxes(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node, opName string) []int {
	var axes []int
	if len(inputs) > 1 && inputs[1] != nil {
		if !inputs[1].DType().IsInt() {
			exceptions.Panicf("axes must be integer, got %s for node %s", inputs[1].DType(), nodeToString(node))
		}

		axesT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
		if err != nil {
			panic(errors.WithMessagef(err, "while converting 'axes' for node %s", nodeToString(node)))
		}
		axes = tensorToInts(axesT)
	}

	axesFromAttr := getIntsAttrOr(node, "axes", nil)
	if len(axesFromAttr) > 0 {
		if len(axes) > 0 {
			exceptions.Panicf("%s(operand, [axes]): axes input and axes attribute cannot be used together for node %s", opName, nodeToString(node))
		}
		axes = axesFromAttr
	}

	return axes
}

// convertArgMax converts a ONNX ArgMax node to a GoMLX node.
// ArgMax computes the indices of the maximum elements along a specified axis.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__ArgMax.html
func convertArgMax(node *protos.NodeProto, inputs []*Node) *Node {
	operand := inputs[0]
	axis := getIntAttrOr(node, "axis", 0)
	keepDims := getBoolAttrOr(node, "keepdims", true)
	selectLastIndex := getBoolAttrOr(node, "select_last_index", false)
	return convertArgReduceOp(operand, axis, keepDims, selectLastIndex, ArgMax)
}

// convertArgMin converts a ONNX ArgMin node to a GoMLX node.
// ArgMin computes the indices of the minimum elements along a specified axis.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__ArgMin.html
func convertArgMin(node *protos.NodeProto, inputs []*Node) *Node {
	operand := inputs[0]
	axis := getIntAttrOr(node, "axis", 0)
	keepDims := getBoolAttrOr(node, "keepdims", true)
	selectLastIndex := getBoolAttrOr(node, "select_last_index", false)
	return convertArgReduceOp(operand, axis, keepDims, selectLastIndex, ArgMin)
}
