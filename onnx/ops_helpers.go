package onnx

// This file contains helper functions used across ONNX operator implementations
// to reduce code duplication and improve maintainability.

import (
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/onnx-gomlx/internal/protos"
)

// getAndAdjustAxis extracts an axis attribute from a node and adjusts it for the operand's rank.
// This handles negative axis values (counting from the end).
func getAndAdjustAxis(node *protos.NodeProto, operand *Node, attrName string, defaultValue int) int {
	axis := getIntAttrOr(node, attrName, defaultValue)
	return AdjustAxisToOperandRank(operand, axis)
}

// getSpatialAxes returns the spatial axes for a tensor in NCHW format.
// For a 4D tensor [N, C, H, W], returns [2, 3].
// For a 5D tensor [N, C, D, H, W], returns [2, 3, 4].
func getSpatialAxes(x *Node) []int {
	axes := make([]int, x.Rank()-2)
	for i := 2; i < x.Rank(); i++ {
		axes[i-2] = i
	}
	return axes
}

// normalizeWithMeanVariance performs mean-variance normalization on a tensor.
// It computes: (x - mean) / sqrt(variance + epsilon)
// The reduction is performed over the specified axes.
func normalizeWithMeanVariance(x *Node, axes []int, epsilon float64) *Node {
	g := x.Graph()
	dtype := x.DType()

	mean := ReduceAndKeep(x, ReduceMean, axes...)
	centered := Sub(x, mean)
	variance := ReduceAndKeep(Mul(centered, centered), ReduceMean, axes...)
	epsilonNode := Scalar(g, dtype, epsilon)

	return Div(centered, Sqrt(Add(variance, epsilonNode)))
}

// accumulateInputs applies a binary operation to accumulate all inputs into a single result.
// Inputs are first broadcast to a common shape following ONNX semantics.
func accumulateInputs(inputs []*Node, op func(*Node, *Node) *Node) *Node {
	if len(inputs) == 1 {
		return inputs[0]
	}

	inputs = onnxBroadcastToCommonShape(inputs)
	result := inputs[0]
	for i := 1; i < len(inputs); i++ {
		result = op(result, inputs[i])
	}
	return result
}

// argReduceFn is a function type for argmax/argmin operations.
type argReduceFn func(*Node, int) *Node

// convertArgReduceOp is a generic implementation for ArgMax and ArgMin operators.
// It handles the common logic for both operations including select_last_index and keepdims.
func convertArgReduceOp(operand *Node, axis int, keepDims, selectLastIndex bool, argFn argReduceFn) *Node {
	axis = AdjustAxisToOperandRank(operand, axis)

	var result *Node
	if selectLastIndex {
		// To select last index, reverse the axis, find argmax/min, then compute correct index
		reversed := Reverse(operand, axis)
		result = argFn(reversed, axis)
		dimSize := Scalar(operand.Graph(), operand.DType(), operand.Shape().Dim(axis))
		result = Sub(Sub(dimSize, Scalar(operand.Graph(), result.DType(), 1)), result)
	} else {
		result = argFn(operand, axis)
	}

	if keepDims {
		result = ExpandAxes(result, axis)
	}

	return result
}

// reshapeTransposeReshape applies a reshape-transpose-reshape pipeline.
// This is commonly used in spatial rearrangement operations like DepthToSpace and SpaceToDepth.
func reshapeTransposeReshape(x *Node, reshapeDims1, transposePerm, reshapeDims2 []int) *Node {
	reshaped := Reshape(x, reshapeDims1...)
	transposed := TransposeAllAxes(reshaped, transposePerm...)
	return Reshape(transposed, reshapeDims2...)
}

// extract4DShape extracts N, C, H, W dimensions from a 4D tensor.
func extract4DShape(x *Node) (n, c, h, w int) {
	dims := x.Shape().Dimensions
	return dims[0], dims[1], dims[2], dims[3]
}

// thresholdActivation applies a threshold-based activation function.
// condition: function that returns a boolean node indicating when to use trueValue
// trueValue: function that computes the result when condition is true
// falseValue: function that computes the result when condition is false
func thresholdActivation(x *Node, condition func(*Node) *Node, trueValue, falseValue func(*Node) *Node) *Node {
	return Where(condition(x), trueValue(x), falseValue(x))
}
