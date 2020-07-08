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


#include "cutlass/array.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
  bool SplitKSerial               ///! If true, code supporting split-K via serial reduction is enabled.
>
struct Conv {
  constexpr static int kAxes = 2;


  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static bool const kSplitKSerial = SplitKSerial;

  // int x__ = Debugx<100, Mma, Epilogue, OutputOp, ThreadblockSwizzle>::f();

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Parameters structure
  struct Params {
    cutlass::Array<int, kAxes> matrix_dims;
    cutlass::Array<int, kAxes> window_sizes;
    int channels;
    cutlass::gemm::GemmCoord problem_size_inner;
    cutlass::gemm::GemmCoord problem_size_outer;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorA::TensorRef ref_A;
    typename Mma::IteratorB::Params params_B;
    typename Mma::IteratorB::TensorRef ref_B;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::TensorRef ref_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::OutputTileIterator::TensorRef ref_D;
    typename OutputOp::Params output_op;
    int *semaphore;
    int gemm_k_iterations; // TODO(klecki): LOL, this is not set
    // int gemm_k_size;
    int gemm_k_size_inner;
    int gemm_k_size_outer;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): semaphore(0) {} //, gemm_k_iterations(0), gemm_k_size(0) { }

    CUTLASS_HOST_DEVICE
    Params(
      const Array<int, kAxes> &matrix_dims_,
      const Array<int, kAxes> &window_sizes_,
      int channels_,
      cutlass::gemm::GemmCoord const & grid_tiled_shape,
      typename Mma::IteratorA::TensorRef ref_A,
      typename Mma::IteratorB::TensorRef ref_B,
      typename Epilogue::OutputTileIterator::TensorRef ref_C,
      typename Epilogue::OutputTileIterator::TensorRef ref_D,
      typename OutputOp::Params output_op = typename OutputOp::Params(),
      int *workspace = nullptr
    ):
      matrix_dims(matrix_dims_),
      window_sizes(window_sizes_),
      channels(channels_),
      // TODO(klecki): not the most obvious place for problem size calculation
      // Left matrix is H x WC = M x K, virtual right is WC x WC = K x N, hence M, N, K := H, WC, WC
      problem_size_inner(matrix_dims[0], matrix_dims[1] * channels, matrix_dims[1] * channels),
      // Right matrix is H x WC = K x N, left is H x H = M x K; hence M, N, K := H, WC, H
      problem_size_outer(matrix_dims[0], matrix_dims[1] * channels, matrix_dims[0]),
      grid_tiled_shape(grid_tiled_shape),
      params_A(ref_A.layout()),
      ref_A(ref_A),
      params_B(ref_B.layout()),
      ref_B(ref_B),
      params_C(ref_C.layout()),
      ref_C(ref_C),
      params_D(ref_D.layout()),
      ref_D(ref_D),
      output_op(output_op) {

      // int total_gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
      // int gemm_k_iterations = (total_gemm_k_iterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();

      gemm_k_size_inner = calc_gemm_k_size(problem_size_inner, grid_tiled_shape);
      gemm_k_size_outer = calc_gemm_k_size(problem_size_outer, grid_tiled_shape);

      semaphore = workspace;
    }

    CUTLASS_HOST_DEVICE
    static int calc_gemm_k_size(gemm::GemmCoord const &problem_size, GemmCoord const &grid_tiled_shape) {
      int total_gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
      int gemm_k_iterations = (total_gemm_k_iterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();
      return gemm_k_iterations * Mma::Shape::kK;
    }

  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Conv() { }

  /// Determines whether kernel satisfies alignment
    static Status can_implement(
      cutlass::gemm::GemmCoord const & problem_size,
      typename Mma::IteratorA::TensorRef ref_A,
      typename Mma::IteratorB::TensorRef ref_B,
      typename Epilogue::OutputTileIterator::TensorRef ref_C,
      typename Epilogue::OutputTileIterator::TensorRef ref_D) {

    static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    if (!TensorRef_aligned(ref_A, kAlignmentA)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_B, kAlignmentB)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_C, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_D, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if ((problem_size.m() % kAlignmentA) || (problem_size.k() % kAlignmentA) ||
      (problem_size.n() % kAlignmentB) || (problem_size.k() % kAlignmentB) ||
      (problem_size.m() % kAlignmentC) || (problem_size.n() % kAlignmentC)) {

      return Status::kErrorMisalignedOperand;
    }

    return Status::kSuccess;
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle; // todo(klecki): build TB swizzle for out of bounds accesses

    cutlass::gemm::GemmCoord threadblock_tile_offset = threadblock_swizzle.get_tile_offset();
    // todo(klecki): here we know the actual tile!!! (global tile)
    PRINT_IF
      printf("kernel::Conv::operator() threadIdx: (%d, %d, %d), threadblock_tile_offset mnk:(%d, %d, %d), params.grid_tiled_shape: (%d, %d, %d) \n",
      threadIdx.x, threadIdx.y, threadIdx.z, threadblock_tile_offset.m(), threadblock_tile_offset.n(), threadblock_tile_offset.k(),
      params.grid_tiled_shape.m(), params.grid_tiled_shape.n(), params.grid_tiled_shape.k());

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }

    // Compute initial location in logical coordinates // KL: all threads have the same value here
    // we need to make this "more virtual"
    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.k() * params.gemm_k_size_inner,
    };

    cutlass::MatrixCoord tb_offset_B{
      threadblock_tile_offset.k() * params.gemm_k_size_inner,
      threadblock_tile_offset.n() * Mma::Shape::kN
    };
    // todo(klecki): and here are the tile coordinates, woohoo
    PRINT_IF
      printf("kernel::Conv::operator() threadIdx: (%d, %d, %d), A  row, col:(%d, %d), B row, col: (%d, %d) \n",
      threadIdx.x, threadIdx.y, threadIdx.z, tb_offset_A.row(), tb_offset_A.column(), tb_offset_B.row(), tb_offset_B.column());

    // Problem size is a function of threadblock index in the K dimension
    int problem_size_k = min(
      params.problem_size_inner.k(),
      (threadblock_tile_offset.k() + 1) * params.gemm_k_size_inner);

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;


    PRINT_IF
      printf("kernel::Conv::operator() threadIdx: (%d, %d, %d), problem_size_k: %d, gemm_k_iterations: %d \n",
        threadIdx.x, threadIdx.y, threadIdx.z, problem_size_k, gemm_k_iterations);

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    // Global mem iterators
    typename Mma::IteratorA iterator_A(
      params.params_A,
      params.ref_A.data(),
      {params.problem_size_inner.m(), problem_size_k},
      thread_idx,
      tb_offset_A);


    // Construct the windows:
    // TensorRef
    auto window = shared_storage.main_loop.operand_Window_ref();

    // TODO(klecki):
    // eithere compute the window directly in SMEM or get it from GMEM
    int window_size = 17;
    int radius = window_size / 2;
    for (int i = thread_idx; i < window_size; i++) {
      if (i < radius) {
        window.data()[i] = i;
      } else if (i > radius) {
        window.data()[i] = window_size - 1 - i;
      } else {
        window.data()[i] = 100;
      }
    }
    __syncthreads();
    // PRINT_IF
    //   for (int i = 0; i < 256; i++) {
    //     printf("window %d: %f\n", i, window.data()[i]);
    //   }


    typename Mma::IteratorB iterator_B(
      params.params_B,
      window.data(),
      {problem_size_k, params.problem_size_inner.n()}, // TODO(klecki) sizes etc are dummy
      thread_idx,
      tb_offset_B);

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = __shfl_sync(0x1f, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);



    typename Mma::FragmentC accumulators;

    accumulators.clear();

    if (!kSplitKSerial || gemm_k_iterations > 0) {
      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);
    }

    //
    // Epilogue
    //

    OutputOp output_op(params.output_op);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset = threadblock_swizzle.get_tile_offset();

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.n() * Mma::Shape::kN
    );

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    // Construct the semaphore.
    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    // If performing a reduction via split-K, fetch the initial synchronization
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {

      // Fetch the synchronization lock initially but do not block.
      semaphore.fetch();

      // Indicate which position in a serial reduction the output operator is currently updating
      output_op.set_k_partition(threadblock_tile_offset.k());
    }

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C(
      params.params_C,
      params.ref_C.data(),
      params.problem_size_inner.mn(),
      thread_idx,
      threadblock_offset
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
      params.params_D,
      params.ref_D.data(),
      params.problem_size_inner.mn(),
      thread_idx,
      threadblock_offset
    );

    Epilogue epilogue(
      shared_storage.epilogue,
      thread_idx,
      warp_idx,
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {

      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_offset.k()) {
        iterator_C = iterator_D;
      }

      semaphore.wait(threadblock_tile_offset.k());

      __threadfence();
    }

    // Execute the epilogue operator to update the destination tensor.
    epilogue(output_op, iterator_D, accumulators, iterator_C);

    //
    // Release the semaphore
    //

    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {

      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        lock = threadblock_tile_offset.k() + 1;
      }

      __threadfence();
      semaphore.release(lock);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

