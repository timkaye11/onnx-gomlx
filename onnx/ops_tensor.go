package onnx

// This file implements ONNX tensor manipulation operators.

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/onnx-gomlx/internal/protos"
	"github.com/pkg/errors"
)

// convertSize converts a ONNX Size node to a GoMLX node.
// Size returns the total number of elements in the input tensor.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Size.html
func convertSize(inputs []*Node) *Node {
	x := inputs[0]
	size := x.Shape().Size()
	return Const(x.Graph(), int64(size))
}

// convertOneHot converts a ONNX OneHot node to a GoMLX node.
// OneHot produces a one-hot tensor based on inputs.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__OneHot.html
func convertOneHot(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	indices := inputs[0]
	// depth is the second input (size of one-hot dimension)
	// values is the third input ([off_value, on_value])

	axis := getIntAttrOr(node, "axis", -1)

	// Materialize depth to get the static value
	depthT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
	if err != nil {
		panic(errors.WithMessagef(err, "while converting 'depth' for node %s", nodeToString(node)))
	}
	depth := tensorToInts(depthT)[0]

	// Get values tensor (off_value, on_value)
	values := inputs[2]
	// Extract off_value and on_value
	offValue := Slice(values, AxisRange(0, 1))
	onValue := Slice(values, AxisRange(1, 2))
	offValue = Reshape(offValue) // Convert to scalar
	onValue = Reshape(onValue)   // Convert to scalar

	g := indices.Graph()
	dtype := values.DType()

	// Adjust axis
	outputRank := indices.Rank() + 1
	if axis < 0 {
		axis = outputRank + axis
	}

	// Create the output shape
	outputDims := make([]int, outputRank)
	indicesDims := indices.Shape().Dimensions
	for i := 0; i < axis; i++ {
		outputDims[i] = indicesDims[i]
	}
	outputDims[axis] = depth
	for i := axis + 1; i < outputRank; i++ {
		outputDims[i] = indicesDims[i-1]
	}

	// Create iota along the one-hot axis
	iotaShape := shapes.Make(dtypes.Int64, outputDims...)
	iota := Iota(g, iotaShape, axis)

	// Expand indices to match output shape
	expandedIndices := ExpandAxes(indices, axis)
	expandedIndices = BroadcastToDims(expandedIndices, outputDims...)
	expandedIndices = ConvertDType(expandedIndices, dtypes.Int64)

	// Handle negative indices (wrap around)
	depthNode := Scalar(g, dtypes.Int64, depth)
	zero := ScalarZero(g, dtypes.Int64)
	expandedIndices = Where(LessThan(expandedIndices, zero), Add(expandedIndices, depthNode), expandedIndices)

	// Create one-hot by comparing iota with indices
	mask := Equal(iota, expandedIndices)

	// Apply on_value and off_value
	onValue = BroadcastToDims(onValue, outputDims...)
	offValue = BroadcastToDims(offValue, outputDims...)
	return Where(mask, onValue, offValue)
}

// convertEyeLike converts a ONNX EyeLike node to a GoMLX node.
// EyeLike generates a 2D identity matrix with ones on a diagonal.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__EyeLike.html
func convertEyeLike(node *protos.NodeProto, inputs []*Node) *Node {
	x := inputs[0]
	if x.Rank() != 2 {
		exceptions.Panicf("EyeLike requires 2D input, got rank %d", x.Rank())
	}

	k := getIntAttrOr(node, "k", 0)
	dtype := getDTypeAttrOr(node, "dtype", x.DType())

	rows := x.Shape().Dim(0)
	cols := x.Shape().Dim(1)
	g := x.Graph()

	// Create row and column indices
	rowShape := shapes.Make(dtypes.Int32, rows, cols)
	colShape := shapes.Make(dtypes.Int32, rows, cols)
	rowIndices := Iota(g, rowShape, 0)
	colIndices := Iota(g, colShape, 1)

	// Shift column indices by k
	kNode := Scalar(g, dtypes.Int32, k)
	shiftedColIndices := Sub(colIndices, kNode)

	// Create identity: 1 where row == col - k, 0 otherwise
	mask := Equal(rowIndices, shiftedColIndices)
	one := ScalarOne(g, dtype)
	zero := ScalarZero(g, dtype)
	return Where(mask, one, zero)
}

// convertDepthToSpace converts a ONNX DepthToSpace node to a GoMLX node.
// DepthToSpace rearranges data from depth to spatial dimensions.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__DepthToSpace.html
func convertDepthToSpace(node *protos.NodeProto, inputs []*Node) *Node {
	x := inputs[0]
	blocksize := mustGetIntAttr(node, "blocksize")
	mode := getStringAttrOr(node, "mode", "DCR")

	// Input shape: [N, C, H, W]
	if x.Rank() != 4 {
		exceptions.Panicf("DepthToSpace requires 4D input, got rank %d", x.Rank())
	}

	dims := x.Shape().Dimensions
	n, c, h, w := dims[0], dims[1], dims[2], dims[3]

	if c%(blocksize*blocksize) != 0 {
		exceptions.Panicf("DepthToSpace: C (%d) must be divisible by blocksize^2 (%d)", c, blocksize*blocksize)
	}

	newC := c / (blocksize * blocksize)
	newH := h * blocksize
	newW := w * blocksize

	if mode == "DCR" {
		// DCR mode: depth-column-row
		// Reshape to [N, blocksize, blocksize, newC, H, W]
		reshaped := Reshape(x, n, blocksize, blocksize, newC, h, w)
		// Transpose to [N, newC, H, blocksize, W, blocksize]
		transposed := TransposeAllAxes(reshaped, 0, 3, 4, 1, 5, 2)
		// Reshape to [N, newC, newH, newW]
		return Reshape(transposed, n, newC, newH, newW)
	}

	// CRD mode: column-row-depth
	// Reshape to [N, newC, blocksize, blocksize, H, W]
	reshaped := Reshape(x, n, newC, blocksize, blocksize, h, w)
	// Transpose to [N, newC, H, blocksize, W, blocksize]
	transposed := TransposeAllAxes(reshaped, 0, 1, 4, 2, 5, 3)
	// Reshape to [N, newC, newH, newW]
	return Reshape(transposed, n, newC, newH, newW)
}

// convertSpaceToDepth converts a ONNX SpaceToDepth node to a GoMLX node.
// SpaceToDepth rearranges data from spatial dimensions to depth.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__SpaceToDepth.html
func convertSpaceToDepth(node *protos.NodeProto, inputs []*Node) *Node {
	x := inputs[0]
	blocksize := mustGetIntAttr(node, "blocksize")

	// Input shape: [N, C, H, W]
	if x.Rank() != 4 {
		exceptions.Panicf("SpaceToDepth requires 4D input, got rank %d", x.Rank())
	}

	dims := x.Shape().Dimensions
	n, c, h, w := dims[0], dims[1], dims[2], dims[3]

	if h%blocksize != 0 || w%blocksize != 0 {
		exceptions.Panicf("SpaceToDepth: H (%d) and W (%d) must be divisible by blocksize (%d)", h, w, blocksize)
	}

	newC := c * blocksize * blocksize
	newH := h / blocksize
	newW := w / blocksize

	// Reshape to [N, C, H/blocksize, blocksize, W/blocksize, blocksize]
	reshaped := Reshape(x, n, c, newH, blocksize, newW, blocksize)
	// Transpose to [N, blocksize, blocksize, C, H/blocksize, W/blocksize]
	transposed := TransposeAllAxes(reshaped, 0, 3, 5, 1, 2, 4)
	// Reshape to [N, newC, newH, newW]
	return Reshape(transposed, n, newC, newH, newW)
}

// convertGatherND converts a ONNX GatherND node to a GoMLX node.
// GatherND gathers slices from data by indexing.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__GatherND.html
func convertGatherND(node *protos.NodeProto, inputs []*Node) *Node {
	data := inputs[0]
	indices := inputs[1]
	batchDims := getIntAttrOr(node, "batch_dims", 0)

	if batchDims != 0 {
		exceptions.Panicf("GatherND with batch_dims != 0 is not yet supported")
	}

	// indices shape: [..., index_depth]
	// where index_depth is the number of dimensions to index into data
	indicesShape := indices.Shape()
	indexDepth := indicesShape.Dim(indicesShape.Rank() - 1)

	if indexDepth > data.Rank() {
		exceptions.Panicf("GatherND: index depth (%d) cannot exceed data rank (%d)", indexDepth, data.Rank())
	}

	// Use GoMLX Gather which expects indices of shape [..., numIndices]
	return Gather(data, indices)
}

// convertScatterElements converts a ONNX ScatterElements node to a GoMLX node.
// ScatterElements scatters individual elements from updates into data based on indices.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__ScatterElements.html
func convertScatterElements(node *protos.NodeProto, inputs []*Node) *Node {
	data := inputs[0]
	indices := inputs[1]
	updates := inputs[2]

	axis := getIntAttrOr(node, "axis", 0)
	reduction := getStringAttrOr(node, "reduction", "none")
	axis = AdjustAxisToOperandRank(data, axis)

	if data.Rank() != indices.Rank() || data.Rank() != updates.Rank() {
		exceptions.Panicf("ScatterElements: data, indices, and updates must have the same rank")
	}

	// Build full indices for all dimensions
	indicesDims := indices.Shape().Dimensions
	indicesSize := indices.Shape().Size()
	g := data.Graph()

	fullIndicesParts := make([]*Node, 0, data.Rank())
	iotaShape := indices.Shape().Clone()
	iotaShape.Dimensions = append(iotaShape.Dimensions, 1)

	for dimIdx := range data.Rank() {
		var part *Node
		if dimIdx == axis {
			// On the scatter axis, use the provided indices
			part = Reshape(indices, indicesSize, 1)
		} else {
			// On other axes, use iota indices
			part = Iota(g, iotaShape, dimIdx)
			part = Reshape(part, indicesSize, 1)
		}
		fullIndicesParts = append(fullIndicesParts, part)
	}
	fullIndices := Concatenate(fullIndicesParts, -1)
	flatUpdates := Reshape(updates, indicesSize)

	switch reduction {
	case "none":
		return ScatterUpdate(data, fullIndices, flatUpdates)
	case "add":
		return ScatterSum(data, fullIndices, flatUpdates)
	case "max":
		return ScatterMax(data, fullIndices, flatUpdates)
	case "min":
		return ScatterMin(data, fullIndices, flatUpdates)
	case "mul":
		// ScatterMul not directly available, implement using multiple operations
		exceptions.Panicf("ScatterElements with reduction='mul' is not yet supported")
	}
	exceptions.Panicf("ScatterElements: unknown reduction mode %q", reduction)
	return nil
}

// convertCompress converts a ONNX Compress node to a GoMLX node.
// Compress selects slices from an input tensor along a given axis based on a condition.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__Compress.html
func convertCompress(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	data := inputs[0]
	condition := inputs[1]

	// Compress requires knowing which elements to select at graph construction time
	// if condition is dynamic, we need to materialize it
	condT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
	if err != nil {
		panic(errors.WithMessagef(err, "Compress requires static condition, failed to materialize for node %s", nodeToString(node)))
	}

	// Get axis - if not specified, data is flattened
	axisAttr := getNodeAttr(node, "axis", false)

	if axisAttr == nil {
		// Flatten data
		data = Reshape(data, data.Shape().Size())
	}

	axis := getIntAttrOr(node, "axis", 0)
	axis = AdjustAxisToOperandRank(data, axis)

	// Extract indices where condition is true
	condVals := tensorToInts(condT)
	var selectedIndices []int
	for i, v := range condVals {
		if v != 0 {
			selectedIndices = append(selectedIndices, i)
		}
	}

	if len(selectedIndices) == 0 {
		// Return empty tensor along the axis
		newDims := data.Shape().Dimensions
		newDims[axis] = 0
		return Zeros(data.Graph(), shapes.Make(data.DType(), newDims...))
	}

	// Gather along the axis
	g := data.Graph()
	indicesNode := Const(g, selectedIndices)
	return onnxGather(data, indicesNode, axis)
}

// convertReverseSequence converts a ONNX ReverseSequence node to a GoMLX node.
// ReverseSequence reverses variable length slices along specified axes.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__ReverseSequence.html
func convertReverseSequence(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	data := inputs[0]
	sequenceLens := inputs[1]

	batchAxis := getIntAttrOr(node, "batch_axis", 1)
	timeAxis := getIntAttrOr(node, "time_axis", 0)

	batchAxis = AdjustAxisToOperandRank(data, batchAxis)
	timeAxis = AdjustAxisToOperandRank(data, timeAxis)

	// ReverseSequence with variable lengths is complex and may require
	// materializing sequence_lens if they're constant
	seqLensT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
	if err != nil {
		panic(errors.WithMessagef(err, "ReverseSequence requires static sequence_lens for node %s", nodeToString(node)))
	}

	seqLens := tensorToInts(seqLensT)
	g := data.Graph()
	dims := data.Shape().Dimensions
	batchSize := dims[batchAxis]
	timeSize := dims[timeAxis]

	if len(seqLens) != batchSize {
		exceptions.Panicf("ReverseSequence: sequence_lens length (%d) must match batch dimension (%d)", len(seqLens), batchSize)
	}

	// For each batch, reverse the first sequence_lens[batch] elements along time axis
	// This requires building a gather indices tensor

	// Create indices tensor [batch, time] for the time axis
	// that represents the reversed indices based on sequence lengths

	// Build result by iterating over batches and slicing
	// This is a simplified implementation that works for 2D case
	if data.Rank() != 2 {
		exceptions.Panicf("ReverseSequence currently only supports 2D tensors")
	}

	// Create reversed index tensor
	resultParts := make([]*Node, batchSize)
	for b := 0; b < batchSize; b++ {
		seqLen := seqLens[b]
		if seqLen > timeSize {
			seqLen = timeSize
		}

		// Get the batch slice
		var batchSlice *Node
		if batchAxis == 0 {
			batchSlice = Slice(data, AxisRange(b, b+1), AxisRange())
		} else {
			batchSlice = Slice(data, AxisRange(), AxisRange(b, b+1))
		}
		batchSlice = Reshape(batchSlice, timeSize)

		// Reverse first seqLen elements
		if seqLen > 0 {
			// Get the part to reverse
			toReverse := Slice(batchSlice, AxisRange(0, seqLen))
			reversed := Reverse(toReverse, 0)

			// Get the part to keep
			if seqLen < timeSize {
				toKeep := Slice(batchSlice, AxisRange(seqLen, timeSize))
				batchSlice = Concatenate([]*Node{reversed, toKeep}, 0)
			} else {
				batchSlice = reversed
			}
		}
		resultParts[b] = ExpandAxes(batchSlice, batchAxis)
	}

	return Concatenate(resultParts, batchAxis)
}

// convertFlatten converts a ONNX Flatten node to a GoMLX node.
// Already implemented in ops.go but included here for completeness of tensor ops.

// convertNonZero converts a ONNX NonZero node to a GoMLX node.
// NonZero returns the indices of non-zero elements.
// Note: This requires knowing the number of non-zero elements at graph construction time,
// which means the input must be a constant expression.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__NonZero.html
func convertNonZero(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) *Node {
	// NonZero is tricky because the output shape depends on the input values,
	// not just the input shape. We need to materialize the input to count non-zeros.
	inputT, err := m.materializeConstantExpression(node.Input[0], convertedOutputs)
	if err != nil {
		panic(errors.WithMessagef(err, "NonZero requires static input for node %s", nodeToString(node)))
	}

	// Count non-zero elements and get their indices
	inputShape := inputs[0].Shape()
	rank := inputShape.Rank()

	// Flatten the input and find non-zero positions
	vals := tensorToInts(inputT)
	var nonZeroPositions []int
	for i, v := range vals {
		if v != 0 {
			nonZeroPositions = append(nonZeroPositions, i)
		}
	}

	numNonZero := len(nonZeroPositions)
	if numNonZero == 0 {
		// Return empty tensor of shape [rank, 0]
		return Zeros(inputs[0].Graph(), shapes.Make(dtypes.Int64, rank, 0))
	}

	// Convert flat positions to multi-dimensional indices
	// Output shape is [rank, numNonZero]
	indices := make([][]int64, rank)
	for i := range rank {
		indices[i] = make([]int64, numNonZero)
	}

	for posIdx, flatPos := range nonZeroPositions {
		// Convert flat position to multi-dimensional index
		remaining := flatPos
		for axis := rank - 1; axis >= 0; axis-- {
			dim := inputShape.Dim(axis)
			indices[axis][posIdx] = int64(remaining % dim)
			remaining /= dim
		}
	}

	// Build the result tensor
	g := inputs[0].Graph()
	resultParts := make([]*Node, rank)
	for axis := range rank {
		resultParts[axis] = Const(g, indices[axis])
		resultParts[axis] = Reshape(resultParts[axis], 1, numNonZero)
	}

	return Concatenate(resultParts, 0)
}

// convertTopK converts a ONNX TopK node to a GoMLX node.
// TopK retrieves the top-K largest or smallest elements along a specified axis.
//
// See ONNX documentation in:
// https://onnx.ai/onnx/operators/onnx__TopK.html
func convertTopK(m *Model, convertedOutputs map[string]*Node, node *protos.NodeProto, inputs []*Node) ([]*Node, error) {
	x := inputs[0]

	// Get K from input[1]
	kT, err := m.materializeConstantExpression(node.Input[1], convertedOutputs)
	if err != nil {
		return nil, errors.WithMessagef(err, "TopK requires static k for node %s", nodeToString(node))
	}
	k := tensorToInts(kT)[0]

	axis := getIntAttrOr(node, "axis", -1)
	largest := getBoolAttrOr(node, "largest", true)
	sorted := getBoolAttrOr(node, "sorted", true)
	_ = sorted // GoMLX TopK always returns sorted results

	axis = AdjustAxisToOperandRank(x, axis)

	var values, indices *Node
	if largest {
		values, indices = TopK(x, k, axis)
	} else {
		// For smallest, negate, get top-k, then negate back
		negX := Neg(x)
		values, indices = TopK(negX, k, axis)
		values = Neg(values)
	}

	return []*Node{values, indices}, nil
}
