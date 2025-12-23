package onnx

// This file implements miscellaneous ONNX operators.

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/onnx-gomlx/internal/protos"
)

// convertDropout converts a ONNX Dropout node to a GoMLX node.
// In inference mode (which is what we primarily support), Dropout is a no-op.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Dropout.html
func convertDropout(convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	// In inference mode, dropout is identity
	x := inputs[0]

	// Dropout can have multiple outputs: (output, mask)
	// For inference, output = input and mask is all true
	if len(node.Output) > 1 && node.Output[1] != "" {
		// Create a mask of all true values
		maskShape := x.Shape()
		mask := Ones(x.Graph(), shapes.Make(dtypes.Bool, maskShape.Dimensions...))
		convertedOutputs[node.Output[1]] = mask
	}

	return x
}

// convertLRN converts a ONNX LRN node to a GoMLX node.
// LRN (Local Response Normalization) normalizes over local input regions.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__LRN.html
func convertLRN(node *protos.NodeProto, inputs []*Node) *Node {
	x := inputs[0]

	alpha := getFloatAttrOr(node, "alpha", 0.0001)
	beta := getFloatAttrOr(node, "beta", 0.75)
	bias := getFloatAttrOr(node, "bias", 1.0)
	size := mustGetIntAttr(node, "size")

	// LRN formula: y = x / (bias + (alpha/size) * sum(x^2))^beta
	// where the sum is over the local region of 'size' channels

	// Input shape: [N, C, H, W] (or [N, C, ...] for other spatial dims)
	if x.Rank() < 3 {
		exceptions.Panicf("LRN requires at least 3D input, got rank %d", x.Rank())
	}

	g := x.Graph()
	dtype := x.DType()
	alphaNode := Scalar(g, dtype, alpha)
	betaNode := Scalar(g, dtype, beta)
	biasNode := Scalar(g, dtype, bias)
	sizeNode := Scalar(g, dtype, float64(size))

	// Square the input
	squared := Mul(x, x)

	// Sum over local region using convolution with ones kernel
	// This is a 1D convolution along the channel axis
	numChannels := x.Shape().Dim(1)
	halfSize := size / 2

	// Pad the channel dimension
	padBefore := halfSize
	padAfter := size - 1 - halfSize

	// Create a padded version for sliding sum
	// We need to sum 'size' neighboring channels
	var sumSquared *Node

	if x.Rank() == 4 {
		// Standard 4D case [N, C, H, W]
		// Pad along channel axis
		paddedSquared := Pad(squared, padBefore, padAfter, 1) // Pad channel axis

		// Use cumulative sum and difference to get sliding window sum
		cumSum := CumSum(paddedSquared, 1)
		// Sum of window [i, i+size) = cumSum[i+size] - cumSum[i]

		// Slice to get the shifted cumsum
		dims := paddedSquared.Shape().Dimensions
		sumSquared = Sub(
			Slice(cumSum, AxisRange(), AxisRange(size, dims[1]), AxisRange(), AxisRange()),
			Slice(cumSum, AxisRange(), AxisRange(0, dims[1]-size), AxisRange(), AxisRange()),
		)

		// Handle the first 'size' elements specially
		// Actually, with proper padding, we can get the right shape directly
		// Recalculate: after padding, channel dim is numChannels + size - 1
		// After sliding window sum, we want numChannels outputs
		sumSquared = Slice(cumSum, AxisRange(), AxisRange(size-1, numChannels+size-1), AxisRange(), AxisRange())
		// Adjust for cumsum starting at 0
		shifted := Pad(cumSum, 0, 0, 0)
		// Actually, let's use a simpler approach with a 1D convolution

		// Simpler approach: use a 1x1x1 convolution with size-length kernel
		// For now, use a loop-based approach (less efficient but clearer)
		parts := make([]*Node, numChannels)
		for c := 0; c < numChannels; c++ {
			startC := c - halfSize
			endC := c + halfSize + 1
			if startC < 0 {
				startC = 0
			}
			if endC > numChannels {
				endC = numChannels
			}
			// Sum squared values over the channel window
			channelSlice := Slice(squared, AxisRange(), AxisRange(startC, endC), AxisRange(), AxisRange())
			parts[c] = ReduceSum(channelSlice, 1)
		}
		sumSquared = Concatenate(parts, 1)
	} else {
		// Generic case - use simple loop approach
		parts := make([]*Node, numChannels)
		for c := 0; c < numChannels; c++ {
			startC := c - halfSize
			endC := c + halfSize + 1
			if startC < 0 {
				startC = 0
			}
			if endC > numChannels {
				endC = numChannels
			}
			channelSlice := SliceAxis(squared, 1, AxisRange(startC, endC))
			parts[c] = ReduceAndKeep(channelSlice, ReduceSum, 1)
		}
		sumSquared = Concatenate(parts, 1)
	}

	// Apply LRN formula: y = x / (bias + (alpha/size) * sum_squared)^beta
	scaledSum := Mul(Div(alphaNode, sizeNode), sumSquared)
	denominator := Pow(Add(biasNode, scaledSum), betaNode)
	return Div(x, denominator)
}

// convertInstanceNormalization converts a ONNX InstanceNormalization node to a GoMLX node.
// InstanceNormalization normalizes each instance independently across spatial dimensions.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__InstanceNormalization.html
func convertInstanceNormalization(node *protos.NodeProto, inputs []*Node) *Node {
	x := inputs[0]
	scale := inputs[1]
	bias := inputs[2]

	epsilon := getFloatAttrOr(node, "epsilon", 1e-5)

	// Input shape: [N, C, H, W, ...] (at least 3D)
	if x.Rank() < 3 {
		exceptions.Panicf("InstanceNormalization requires at least 3D input, got rank %d", x.Rank())
	}

	// Normalize over spatial dimensions (all except batch and channel)
	spatialAxes := getSpatialAxes(x)

	// Normalize: (x - mean) / sqrt(variance + epsilon)
	normalized := normalizeWithMeanVariance(x, spatialAxes, float64(epsilon))

	// Apply scale and bias
	// Scale and bias have shape [C], need to broadcast to [1, C, 1, 1, ...]
	numOnes := x.Rank() - 2
	for i := 0; i < numOnes; i++ {
		scale = ExpandAxes(scale, -1)
		bias = ExpandAxes(bias, -1)
	}
	scale = ExpandAxes(scale, 0) // Add batch dimension
	bias = ExpandAxes(bias, 0)

	return Add(Mul(scale, normalized), bias)
}

// convertGlobalMaxPool converts a ONNX GlobalMaxPool node to a GoMLX node.
// GlobalMaxPool applies max pooling over the entire spatial extent.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__GlobalMaxPool.html
func convertGlobalMaxPool(inputs []*Node) *Node {
	x := inputs[0]

	// Input shape: [N, C, H, W, ...] (at least 3D)
	if x.Rank() < 3 {
		exceptions.Panicf("GlobalMaxPool requires at least 3D input, got rank %d", x.Rank())
	}

	// ReduceMax over spatial dimensions, keeping dims for compatibility
	return ReduceAndKeep(x, ReduceMax, getSpatialAxes(x)...)
}

// convertGlobalLpPool converts a ONNX GlobalLpPool node to a GoMLX node.
// GlobalLpPool applies Lp pooling over the entire spatial extent.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__GlobalLpPool.html
func convertGlobalLpPool(node *protos.NodeProto, inputs []*Node) *Node {
	x := inputs[0]
	p := getIntAttrOr(node, "p", 2)

	if x.Rank() < 3 {
		exceptions.Panicf("GlobalLpPool requires at least 3D input, got rank %d", x.Rank())
	}

	g := x.Graph()
	dtype := x.DType()
	pNode := Scalar(g, dtype, float64(p))
	invP := Scalar(g, dtype, 1.0/float64(p))

	// Lp pooling: (sum(|x|^p))^(1/p)
	absPowP := Pow(Abs(x), pNode)
	sumPow := ReduceAndKeep(absPowP, ReduceSum, getSpatialAxes(x)...)
	return Pow(sumPow, invP)
}

// convertLpNormalization converts a ONNX LpNormalization node to a GoMLX node.
// LpNormalization normalizes input tensor using Lp norm along specified axis.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__LpNormalization.html
func convertLpNormalization(node *protos.NodeProto, inputs []*Node) *Node {
	x := inputs[0]
	axis := getIntAttrOr(node, "axis", -1)
	p := getIntAttrOr(node, "p", 2)

	axis = AdjustAxisToOperandRank(x, axis)
	g := x.Graph()
	dtype := x.DType()

	var norm *Node
	if p == 1 {
		// L1 norm: sum(|x|)
		norm = ReduceAndKeep(Abs(x), ReduceSum, axis)
	} else if p == 2 {
		// L2 norm: sqrt(sum(x^2))
		norm = Sqrt(ReduceAndKeep(Mul(x, x), ReduceSum, axis))
	} else {
		// General Lp norm: (sum(|x|^p))^(1/p)
		pNode := Scalar(g, dtype, float64(p))
		invP := Scalar(g, dtype, 1.0/float64(p))
		absPowP := Pow(Abs(x), pNode)
		norm = Pow(ReduceAndKeep(absPowP, ReduceSum, axis), invP)
	}

	// Avoid division by zero
	epsilon := Scalar(g, dtype, 1e-10)
	norm = Max(norm, epsilon)

	return Div(x, norm)
}

// convertMeanVarianceNormalization converts a ONNX MeanVarianceNormalization node to a GoMLX node.
// MeanVarianceNormalization normalizes input tensor to have zero mean and unit variance.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__MeanVarianceNormalization.html
func convertMeanVarianceNormalization(node *protos.NodeProto, inputs []*Node) *Node {
	x := inputs[0]
	axes := getIntsAttrOr(node, "axes", []int{0, 2, 3})

	// Normalize: (x - mean) / sqrt(variance + epsilon)
	return normalizeWithMeanVariance(x, axes, 1e-9)
}

// convertCastLike converts a ONNX CastLike node to a GoMLX node.
// CastLike casts input to the same data type as the target tensor.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__CastLike.html
func convertCastLike(inputs []*Node) *Node {
	x := inputs[0]
	target := inputs[1]
	return ConvertDType(x, target.DType())
}

// convertNeg is already implemented in graph.go as a unary operator

// convertDetNot converts a ONNX Det node to a GoMLX node.
// Det computes the determinant of a square matrix.
// Note: This is a complex operation that may not be directly supported.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Det.html
// func convertDet(inputs []*Node) *Node {
//     // Determinant computation requires LU decomposition or similar
//     // This is not directly available in GoMLX
//     exceptions.Panicf("Det operator is not yet supported")
//     return nil
// }

// convertEinsum converts a ONNX Einsum node to a GoMLX node.
// Einsum performs tensor contraction based on Einstein summation notation.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Einsum.html
func convertEinsum(node *protos.NodeProto, inputs []*Node) *Node {
	equation := getStringAttrOr(node, "equation", "")
	if equation == "" {
		exceptions.Panicf("Einsum requires 'equation' attribute")
	}
	return Einsum(equation, inputs...)
}

// convertSequenceLength converts a ONNX SequenceLength node to a GoMLX node.
// Note: Sequence operations are not fully supported, this is a placeholder.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__SequenceLength.html
// func convertSequenceLength(inputs []*Node) *Node {
//     exceptions.Panicf("SequenceLength operator is not yet supported")
//     return nil
// }
