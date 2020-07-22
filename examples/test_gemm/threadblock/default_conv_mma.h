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

////////////////////////////////////////////////////////////////////////////////
template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for
    typename ArchTag_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_,
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
struct DefaultConvMma;


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
  using UnderlyingMma = DefaultConvMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
				ElementAccumulator, LayoutC, OperatorClass, ArchTag, ThreadblockShape, WarpShape,
				InstructionShape, Stages, Operator, AccumulatorsInRowMajor>;

	 // Define the MmaCore components
  using MmaCore = typename UnderlyingMma::MmaCore;

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


////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output (OperatorClass Simt)
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
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator,
    /// Type of Convolution
    bool InnerConv>
struct DefaultConvMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, layout::RowMajor,
                  arch::OpClassSimt, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, 2, Operator, false, InnerConv> {
  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
      arch::OpClassSimt, 2, Operator>;

  static int const kThreads = MmaCore::kThreads;
  static int const kElementsPerAccess = MmaCore::kElementsPerAccess;
  static int const kInnerConv = InnerConv;

  // Define iterators over tiles from the A operand
  using IteratorA_regular_gmem_ =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA>;

  using IteratorA_outer_conv_smem_ =
      cutlass::transform::threadblock::PositionPredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA>;

  // Define iterators over tiles from the B operand
  using IteratorA = std::conditional_t<kInnerConv, IteratorA_regular_gmem_, IteratorA_outer_conv_smem_>;


  // Define iterators over tiles from the B operand
  using IteratorB_inner_conv_smem_ =
      cutlass::transform::threadblock::PositionPredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB>;

  using IteratorB_regular_gmem_ = cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB>;

  // Define iterators over tiles from the B operand
  using IteratorB = std::conditional_t<kInnerConv, IteratorB_inner_conv_smem_, IteratorB_regular_gmem_>;

//   using TileShapeB = cutlass::layout::PitchLinearShape<MmaCore::Shape::kK, 1>; // we only load one row of window - hmm, it's bad that it's the other way round
//   using TileLayoutB = cutlass::layout::PitchLinear;

//   // in  DefaultMmaCore
//     /// Policy of iterator B
//   using IteratorThreadMapB = transform::PitchLinearStripminedThreadMap<
//     TileShapeB,
//     kThreads,
//     kElementsPerAccess
//   >;


  // ThreadMaps define how threads are mapped to a given tile. The PitchLinearStripminedThreadMap
  // stripmines a pitch-linear tile among a given number of threads, first along the contiguous
  // dimension then along the strided dimension.
  // using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<TileLayoutB, kThreads>;

  // Define the PredicateTileIterator, using TileShape, Element, Layout, and ThreadMap types
  // using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
  //     TileShapeB, ElementB, TileLayoutB, 0, IteratorThreadMapB>; // advance rank TODO???

  // Define the threadblock-scoped pipelined matrix multiply

  // We pass here all the iterators and there is the actual impl of load GMEM->SMEM happening
  using ThreadblockMma = cutlass::gemm::threadblock::ConvMmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      layout::RowMajor, typename MmaCore::MmaPolicy, kInnerConv>;
};

// ////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output (OperatorClass TensorOp)
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
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator,
    /// Type of Convolution
    bool InnerConv
    >
struct DefaultConvMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, layout::RowMajor,
                  arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, 2, Operator, false, InnerConv> {
  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
      arch::OpClassTensorOp, 2, Operator>;

  // Define iterators over tiles from the A operand
//   using IteratorA =
//       cutlass::transform::threadblock::PredicatedTileIterator<
//           cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
//           ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA>;

//   // Define iterators over tiles from the B operand
//   using IteratorB =
//       cutlass::transform::threadblock::PredicatedTileIterator<
//           cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
//           ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB>;



  static int const kInnerConv = InnerConv;

  // Define iterators over tiles from the A operand
  using IteratorA_regular_gmem_ =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA>;

  using IteratorA_outer_conv_smem_ =
      cutlass::transform::threadblock::PositionPredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentB>;

  // Define iterators over tiles from the B operand
  using IteratorA = std::conditional_t<kInnerConv, IteratorA_regular_gmem_, IteratorA_outer_conv_smem_>;


  // Define iterators over tiles from the B operand
  using IteratorB_inner_conv_smem_ =
      cutlass::transform::threadblock::PositionPredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB>;

  using IteratorB_regular_gmem_ = cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB>;

  // Define iterators over tiles from the B operand
  using IteratorB = std::conditional_t<kInnerConv, IteratorB_inner_conv_smem_, IteratorB_regular_gmem_>;


  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = cutlass::gemm::threadblock::ConvMmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      layout::RowMajor, typename MmaCore::MmaPolicy, InnerConv>;
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major output (OperatorClass TensorOp)
template <
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator,
    /// Type of Convolution
    bool InnerConv
    >
struct DefaultConvMma<float, LayoutA, kAlignmentA, float, LayoutB,
                  kAlignmentB, float, layout::RowMajor,
                  arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, 2, Operator, false, InnerConv> {
  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, float, LayoutA, float,
      LayoutB, float, layout::RowMajor, arch::OpClassTensorOp, 2,
      arch::OpMultiplyAddFastF16>;

//   // Define iterators over tiles from the A operand
//   using IteratorA =
//       cutlass::transform::threadblock::PredicatedTileIterator<
//           cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
//           float, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA>;

//   // Define iterators over tiles from the B operand
//   using IteratorB =
//       cutlass::transform::threadblock::PredicatedTileIterator<
//           cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
//           float, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB>;




  static int const kInnerConv = InnerConv;

  // Define iterators over tiles from the A operand
  using IteratorA_regular_gmem_ =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          float, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA>;

  using IteratorA_outer_conv_smem_ =
      cutlass::transform::threadblock::PositionPredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          float, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentB>;

  // Define iterators over tiles from the B operand
  using IteratorA = std::conditional_t<kInnerConv, IteratorA_regular_gmem_, IteratorA_outer_conv_smem_>;


  // Define iterators over tiles from the B operand
  using IteratorB_inner_conv_smem_ =
      cutlass::transform::threadblock::PositionPredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          float, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB>;

  using IteratorB_regular_gmem_ = cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          float, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB>;

  // Define iterators over tiles from the B operand
  using IteratorB = std::conditional_t<kInnerConv, IteratorB_inner_conv_smem_, IteratorB_regular_gmem_>;


  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = cutlass::gemm::threadblock::ConvMmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, float,
      layout::RowMajor, typename MmaCore::MmaPolicy, InnerConv>;
};

// ////////////////////////////////////////////////////////////////////////////////

// /// Specialization for column-major-interleaved output
// template <
//     /// Element type for A matrix operand
//     typename ElementA,
//     /// Layout type for A matrix operand
//     typename LayoutA,
//     /// Access granularity of A matrix in units of elements
//     int kAlignmentA,
//     /// Element type for B matrix operand
//     typename ElementB,
//     /// Layout type for B matrix operand
//     typename LayoutB,
//     /// Access granularity of B matrix in units of elements
//     int kAlignmentB,
//     /// Element type for internal accumulation
//     typename ElementAccumulator,
//     /// Tag indicating architecture to tune for
//     typename OperatorClass,
//     /// Tag indicating architecture to tune for
//     typename ArchTag,
//     /// Threadblock-level tile size (concept: GemmShape)
//     typename ThreadblockShape,
//     /// Warp-level tile size (concept: GemmShape)
//     typename WarpShape,
//     /// Instruction-level tile size (concept: GemmShape)
//     typename InstructionShape,
//     /// Operation performed by GEMM
//     typename Operator,
//     /// Number of Interleaved K
//     int InterleavedK>
// struct DefaultConvMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
//                   kAlignmentB, ElementAccumulator,
//                   layout::ColumnMajorInterleaved<InterleavedK>, OperatorClass,
//                   ArchTag, ThreadblockShape, WarpShape, InstructionShape, 2,
//                   Operator, true> {
//   // Define the MmaCore components
//   using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
//       ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
//       ElementB, LayoutB, ElementAccumulator,
//       layout::ColumnMajorInterleaved<InterleavedK>, OperatorClass, 2, Operator,
//       true>;

//   static_assert(kAlignmentA == 128 / sizeof_bits<ElementA>::value,
//     "Alignment must match thread data map's vector length");

//   static_assert(kAlignmentB ==128 / sizeof_bits<ElementB>::value,
//     "Alignment must match thread data map's vector length");

//   // Define iterators over tiles from the A operand
//   using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
//       cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>, ElementA,
//       LayoutA, 1, typename MmaCore::IteratorThreadMapA>;

//   // Define iterators over tiles from the B operand
//   using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
//       cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>, ElementB,
//       LayoutB, 0, typename MmaCore::IteratorThreadMapB>;

//   // Define the threadblock-scoped pipelined matrix multiply
//   using ThreadblockMma = cutlass::gemm::threadblock::ConvMmaPipelined<
//       typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
//       IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
//       layout::ColumnMajorInterleaved<InterleavedK>,
//       typename MmaCore::MmaPolicy>;
// };

// ////////////////////////////////////////////////////////////////////////////////

// /// Specialization for row-major output
// template <
//     /// Element type for A matrix operand
//     typename ElementA,
//     /// Layout type for A matrix operand
//     typename LayoutA,
//     /// Access granularity of A matrix in units of elements
//     int kAlignmentA,
//     /// Element type for B matrix operand
//     typename ElementB,
//     /// Layout type for B matrix operand
//     typename LayoutB,
//     /// Access granularity of B matrix in units of elements
//     int kAlignmentB,
//     /// Element type for internal accumulation
//     typename ElementAccumulator,
//     /// Tag indicating architecture to tune for
//     typename ArchTag,
//     /// Threadblock-level tile size (concept: GemmShape)
//     typename ThreadblockShape,
//     /// Warp-level tile size (concept: GemmShape)
//     typename WarpShape,
//     /// Instruction-level tile size (concept: GemmShape)
//     typename InstructionShape,
//     /// Number of stages used in the multistage mainloop
//     int Stages,
//     /// Operation perfomed by GEMM
//     typename Operator,
//     /// Type of Convolution
//     bool InnerConv
//     >
// struct DefaultConvMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
//                   kAlignmentB, ElementAccumulator, layout::RowMajor,
//                   arch::OpClassSimt, ArchTag, ThreadblockShape, WarpShape,
//                   InstructionShape, Stages, Operator, false> {
//   // Define the MmaCore components
//   using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
//       ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
//       ElementB, LayoutB, ElementAccumulator, layout::RowMajor, arch::OpClassSimt,
//       Stages, Operator>;

//   // Define iterators over tiles from the A operand
//   using ThreadMapA = typename MmaCore::IteratorThreadMapA;
//   using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
//   using IteratorA =
//       cutlass::transform::threadblock::PredicatedTileAccessIterator<
//           cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
//           ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;

//   // Define iterators over tiles from the B operand
//   using ThreadMapB = typename MmaCore::IteratorThreadMapB;
//   using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
//   using IteratorB =
//       cutlass::transform::threadblock::PredicatedTileAccessIterator<
//           cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
//           ElementB, LayoutB, 0, ThreadMapB, AccessTypeB>;

//   // Define the threadblock-scoped multistage matrix multiply
//   using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage<
//       typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
//       MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
//       MmaCore::kCacheOpB, ElementAccumulator, layout::RowMajor,
//       typename MmaCore::MmaPolicy, Stages>;
// };

// ////////////////////////////////////////////////////////////////////////////////

// /// Specialization for row-major output (OperatorClass TensorOp)
// template <
//     /// Element type for A matrix operand
//     typename ElementA,
//     /// Layout type for A matrix operand
//     typename LayoutA,
//     /// Access granularity of A matrix in units of elements
//     int kAlignmentA,
//     /// Element type for B matrix operand
//     typename ElementB,
//     /// Layout type for B matrix operand
//     typename LayoutB,
//     /// Access granularity of B matrix in units of elements
//     int kAlignmentB,
//     /// Element type for internal accumulation
//     typename ElementAccumulator,
//     /// Tag indicating architecture to tune for
//     typename ArchTag,
//     /// Threadblock-level tile size (concept: GemmShape)
//     typename ThreadblockShape,
//     /// Warp-level tile size (concept: GemmShape)
//     typename WarpShape,
//     /// Instruction-level tile size (concept: GemmShape)
//     typename InstructionShape,
//     /// Number of stages used in the multistage mainloop
//     int Stages,
//     /// Operation perfomed by GEMM
//     typename Operator,
//     /// Type of Convolution
//     bool InnerConv
//     >
// struct DefaultConvMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
//                   kAlignmentB, ElementAccumulator, layout::RowMajor,
//                   arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape,
//                   InstructionShape, Stages, Operator, false> {
//   static cutlass::arch::CacheOperation::Kind const CacheOpA =
//       ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
//           ? cutlass::arch::CacheOperation::Global
//           : cutlass::arch::CacheOperation::Always;

//   static cutlass::arch::CacheOperation::Kind const CacheOpB =
//       ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
//           ? cutlass::arch::CacheOperation::Global
//           : cutlass::arch::CacheOperation::Always;

//   // Define the MmaCore components
//   using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
//       ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
//       ElementB, LayoutB, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
//       Stages, Operator, false, CacheOpA, CacheOpB>;

//   // Define iterators over tiles from the A operand
//   using ThreadMapA = typename MmaCore::IteratorThreadMapA;
//   using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
//   using IteratorA =
//       cutlass::transform::threadblock::PredicatedTileAccessIterator<
//           cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
//           ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;

//   // Define iterators over tiles from the B operand
//   using ThreadMapB = typename MmaCore::IteratorThreadMapB;
//   using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
//   using IteratorB =
//       cutlass::transform::threadblock::PredicatedTileAccessIterator<
//           cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
//           ElementB, LayoutB, 0, ThreadMapB, AccessTypeB>;

//   // Define the threadblock-scoped multistage matrix multiply
//   using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage<
//       typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
//       MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
//       MmaCore::kCacheOpB, ElementAccumulator, layout::RowMajor,
//       typename MmaCore::MmaPolicy, Stages>;
// };

// ////////////////////////////////////////////////////////////////////////////////

// /// Specialization for column-major-interleaved output
// template <
//     /// Element type for A matrix operand
//     typename ElementA,
//     /// Layout type for A matrix operand
//     typename LayoutA,
//     /// Access granularity of A matrix in units of elements
//     int kAlignmentA,
//     /// Element type for B matrix operand
//     typename ElementB,
//     /// Layout type for B matrix operand
//     typename LayoutB,
//     /// Access granularity of B matrix in units of elements
//     int kAlignmentB,
//     /// Element type for internal accumulation
//     typename ElementAccumulator,
//     /// Tag indicating architecture to tune for
//     typename OperatorClass,
//     /// Tag indicating architecture to tune for
//     typename ArchTag,
//     /// Threadblock-level tile size (concept: GemmShape)
//     typename ThreadblockShape,
//     /// Warp-level tile size (concept: GemmShape)
//     typename WarpShape,
//     /// Instruction-level tile size (concept: GemmShape)
//     typename InstructionShape,
//     /// Number of stages used in the multistage mainloop
//     int Stages,
//     /// Operation performed by GEMM
//     typename Operator,
//     /// Number of Interleaved K
//     int InterleavedK>
// struct DefaultConvMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
//                   kAlignmentB, ElementAccumulator,
//                   layout::ColumnMajorInterleaved<InterleavedK>, OperatorClass,
//                   ArchTag, ThreadblockShape, WarpShape, InstructionShape,
//                   Stages, Operator, true> {
//   // Define the MmaCore components
//   using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
//       ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
//       ElementB, LayoutB, ElementAccumulator,
//       layout::ColumnMajorInterleaved<InterleavedK>, OperatorClass, Stages,
//       Operator, true>;

//   // Define iterators over tiles from the A operand
//   using ThreadMapA = typename MmaCore::IteratorThreadMapA;
//   using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
//   using IteratorA =
//       cutlass::transform::threadblock::PredicatedTileAccessIterator<
//           cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
//           ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;

//   // Define iterators over tiles from the B operand
//   using ThreadMapB = typename MmaCore::IteratorThreadMapB;
//   using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
//   using IteratorB =
//       cutlass::transform::threadblock::PredicatedTileAccessIterator<
//           cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
//           ElementB, LayoutB, 0, ThreadMapB, AccessTypeB>;

//   // Define the threadblock-scoped multistage matrix multiply
//   using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage<
//       typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
//       MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
//       MmaCore::kCacheOpB, ElementAccumulator, layout::RowMajor,
//       typename MmaCore::MmaPolicy, Stages>;
// };

// ////////////////////////////////////////////////////////////////////////////////

// /// Specialization for SIMT IDP4A Kernels
// template <
//     /// Layout type for A matrix operand
//     typename LayoutA,
//     /// Access granularity of A matrix in units of elements
//     int kAlignmentA,
//     /// Layout type for B matrix operand
//     typename LayoutB,
//     /// Access granularity of B matrix in units of elements
//     int kAlignmentB,
//     /// Element type for internal accumulation
//     typename ElementAccumulator,
//     /// Tag indicating architecture to tune for
//     typename ArchTag,
//     /// Threadblock-level tile size (concept: GemmShape)
//     typename ThreadblockShape,
//     /// Operation performed by GEMM
//     typename Operator,
//     /// Warp-level tile size (concept: GemmShape)
//     typename WarpShape>
// struct DefaultConvMma<int8_t, LayoutA, kAlignmentA, int8_t, LayoutB, kAlignmentB,
//                   ElementAccumulator, layout::RowMajor, arch::OpClassSimt,
//                   ArchTag, ThreadblockShape, WarpShape, GemmShape<1, 1, 4>, 2,
//                   Operator, false> {
//   using InstructionShape = GemmShape<1, 1, 4>;
//   using ElementA = int8_t;
//   using ElementB = int8_t;
//   using OperatorClass =  arch::OpClassSimt;

//   static const bool transposeA =  cutlass::platform::is_same< LayoutA, layout::ColumnMajor >::value;
//   static const bool transposeB =  cutlass::platform::is_same< LayoutB, layout::RowMajor >::value;

//   // Define the MmaCore components
//   using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
//       ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
//       ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
//       OperatorClass, 2, Operator>;

//   // Define iterators over tiles from the A operand
//   using IteratorA =
//       cutlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
//           cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
//           ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, transposeA>;

//   // Define iterators over tiles from the B operand
//   using IteratorB =
//       cutlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
//           cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
//           ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, transposeB>;

//   // Define the threadblock-scoped pipelined matrix multiply
//   using ThreadblockMma = cutlass::gemm::threadblock::ConvMmaPipelined<
//       typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
//       IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
//       layout::RowMajor, typename MmaCore::MmaPolicy>;
// };

// ////////////////////////////////////////////////////////////////////////////////

// #if defined(CUTLASS_ARCH_WMMA_ENABLED)
// /// Specialization for Wmma TensorOp operator with 2 staged pipeline
// template <
//     ///< Element type for A matrix operand
//     typename ElementA,
//     /// Layout type for A matrix operand
//     typename LayoutA,
//     /// Access granularity of A matrix in units of elements
//     int kAlignmentA,
//     /// Element type for B matrix operand
//     typename ElementB,
//     /// Layout type for B matrix operand
//     typename LayoutB,
//     /// Access granularity of B matrix in units of elements
//     int kAlignmentB,
//     /// Element type for internal accumulation
//     typename ElementAccumulator,
//     /// Layout type for C and D matrix operands
//     typename LayoutC,
//     /// Tag indicating architecture to tune for
//     typename ArchTag,
//     /// Threadblock-level tile size (concept: GemmShape)
//     typename ThreadblockShape,
//     /// Warp-level tile size (concept: GemmShape)
//     typename WarpShape,
//     /// Instruction-level tile size (concept: GemmShape)
//     typename InstructionShape,
//     /// Operation performed by GEMM
//     typename Operator>
// struct DefaultConvMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
//                   kAlignmentB, ElementAccumulator, LayoutC,
//                   arch::OpClassWmmaTensorOp, ArchTag, ThreadblockShape, WarpShape,
//                   InstructionShape, 2, Operator> {
//   // Define the MmaCore components
//   using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
//       ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
//       ElementB, LayoutB, ElementAccumulator, LayoutC,
//       arch::OpClassWmmaTensorOp, 2, Operator>;

//   // Define iterators over tiles from the A operand
//   using IteratorA =
//       cutlass::transform::threadblock::PredicatedTileIterator<
//           cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
//           ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA>;

//   // Define iterators over tiles from the B operand
//   using IteratorB =
//       cutlass::transform::threadblock::PredicatedTileIterator<
//           cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
//           ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB>;

//   // Define the threadblock-scoped pipelined matrix multiply
//   using ThreadblockMma = cutlass::gemm::threadblock::ConvMmaPipelined<
//       typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
//       IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
//       LayoutC, typename MmaCore::MmaPolicy>;
// };

// ////////////////////////////////////////////////////////////////////////////////

// /// Specialization for Wmma TensorOp operator with 1 staged pipeline
// template <
//     ///< Element type for A matrix operand
//     typename ElementA,
//     /// Layout type for A matrix operand
//     typename LayoutA,
//     /// Access granularity of A matrix in units of elements
//     int kAlignmentA,
//     /// Element type for B matrix operand
//     typename ElementB,
//     /// Layout type for B matrix operand
//     typename LayoutB,
//     /// Access granularity of B matrix in units of elements
//     int kAlignmentB,
//     /// Element type for internal accumulation
//     typename ElementAccumulator,
//     /// Layout type for C and D matrix operands
//     typename LayoutC,
//     /// Tag indicating architecture to tune for
//     typename ArchTag,
//     /// Threadblock-level tile size (concept: GemmShape)
//     typename ThreadblockShape,
//     /// Warp-level tile size (concept: GemmShape)
//     typename WarpShape,
//     /// Instruction-level tile size (concept: GemmShape)
//     typename InstructionShape,
//     /// Operation performed by GEMM
//     typename Operator>
// struct DefaultConvMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
//                   kAlignmentB, ElementAccumulator, LayoutC,
//                   arch::OpClassWmmaTensorOp, ArchTag, ThreadblockShape, WarpShape,
//                   InstructionShape, 1, Operator> {
//   // Define the MmaCore components
//   using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
//       ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
//       ElementB, LayoutB, ElementAccumulator, LayoutC,
//       arch::OpClassWmmaTensorOp, 1, Operator>;

//   // Define iterators over tiles from the A operand
//   using IteratorA =
//       cutlass::transform::threadblock::PredicatedTileIterator<
//           cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
//           ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA>;

//   // Define iterators over tiles from the B operand
//   using IteratorB =
//       cutlass::transform::threadblock::PredicatedTileIterator<
//           cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
//           ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB>;

//   // Define the threadblock-scoped singlestage matrix multiply
//   using ThreadblockMma = cutlass::gemm::threadblock::MmaSingleStage<
//       typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
//       IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
//       LayoutC, typename MmaCore::MmaPolicy>;
// };

// ////////////////////////////////////////////////////////////////////////////////
// #endif //CUTLASS_ARCH_WMMA_ENABLED

} // namespace threadblock
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
