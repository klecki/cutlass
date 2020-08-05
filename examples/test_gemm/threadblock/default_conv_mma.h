/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/wmma.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "threadblock/position_predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h"
#include "threadblock/conv_mma_pipelined.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"

#include "cutlass/gemm/threadblock/default_mma.h"


#if defined(CUTLASS_ARCH_WMMA_ENABLED)
#include "cutlass/gemm/threadblock/default_mma_core_wmma.h"
#endif //CUTLASS_ARCH_WMMA_ENABLED

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/// Wraps DefaultMma by adding the PositionPredicatedTile iterator and selection
/// between Inner and Outer Conv.
/// Redirects the appropriate iterators to IteratorA (default for InnerConv)
///  and IteratorB (default for !InnerConv)
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false,
    /// If the convolution is computed in the innermost or outer dimension
    bool InnerConv = true
    >
struct SpecializedConvMma {
  // TODO(klecki): HERE WE CAN SET SOME OTHER TYPES???

  using UnderlyingMma = DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
				ElementAccumulator, LayoutC, OperatorClass, ArchTag, ThreadblockShape, WarpShape,
				InstructionShape, Stages, Operator, AccumulatorsInRowMajor>;


  // Select SMEM iterators that use ElementAccumulator type as storage (and computation)
  // instead of ElementA and ElementB
  using UnderlyingMmaProcessing = DefaultMma<float, LayoutA, kAlignmentA, float, LayoutB, kAlignmentB,
				ElementAccumulator, LayoutC, OperatorClass, ArchTag, ThreadblockShape, WarpShape,
				InstructionShape, Stages, Operator, AccumulatorsInRowMajor>;

	 // Define the MmaCore components
  using MmaCore = typename UnderlyingMmaProcessing::MmaCore;

  static int const kInnerConv = InnerConv;

	// PositionPredicatedTileIterators that build matrix on the fly from SMEM
	using IteratorA_outer_conv_smem_ =
      cutlass::transform::threadblock::PositionPredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA>;

	using IteratorB_inner_conv_smem_ =
      cutlass::transform::threadblock::PositionPredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB>;


  // Define iterators over tiles from the B operand
  using IteratorA = std::conditional_t<kInnerConv, typename UnderlyingMma::IteratorA, IteratorA_outer_conv_smem_>;

  // Define iterators over tiles from the B operand
  using IteratorB = std::conditional_t<kInnerConv, IteratorB_inner_conv_smem_, typename UnderlyingMma::IteratorB>;

	// We pass here all the iterators and there is the actual impl of load GMEM->SMEM happening
	// Overwrite the one from UnderlyingMma
  using ThreadblockMma = cutlass::gemm::threadblock::ConvMmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      layout::RowMajor, typename MmaCore::MmaPolicy, kInnerConv>;
};

} // namespace threadblock
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
