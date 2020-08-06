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
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_iterator.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/util/device_dump.h"

#include "dali/utility.h"

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
  static int const kInnerConv = Mma::kInnerConv;

  /// Linear buffer with window values
  using WindowElement = typename Mma::IteratorWindow::Element;
  using WindowLayout = layout::PitchLinear;
  using WindowRef = TensorRef<WindowElement, layout::PitchLinear>;

  // int x__ = Debugx<100, Mma, Epilogue, OutputOp, ThreadblockSwizzle>::f();

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Parameters structure
  struct SampleParams {
    int channels;
    int window_size;
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    typename Mma::IteratorIn::Params params_In;
    typename Mma::IteratorIn::TensorRef ref_In;
    WindowRef ref_conv_Window;
    typename Mma::IteratorWindow::Params params_Window;  ///< This are parameters for iterator
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::TensorRef ref_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::OutputTileIterator::TensorRef ref_D;
    typename OutputOp::Params output_op;
    int *semaphore; // TODO(klecki): add some handling for generating per sample semaphore?
    int gemm_k_iterations; // TODO(klecki): LOL, this is not set
    int gemm_k_size;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    SampleParams(): semaphore(0) {} //, gemm_k_iterations(0), gemm_k_size(0) { }

    CUTLASS_HOST_DEVICE
    SampleParams(
      int channels,
      cutlass::gemm::GemmCoord const & problem_size,
      cutlass::gemm::GemmCoord const & grid_tiled_shape,
      typename Mma::IteratorIn::TensorRef ref_In,
      WindowRef ref_conv_Window,
      typename Epilogue::OutputTileIterator::TensorRef ref_C,
      typename Epilogue::OutputTileIterator::TensorRef ref_D,
      typename OutputOp::Params output_op = typename OutputOp::Params(),
      int *workspace = nullptr
    ):
      channels(channels),
      window_size(ref_conv_Window.stride(0)),
      problem_size(problem_size), // actual problem size
      grid_tiled_shape(grid_tiled_shape), // the grid used for the run (m, n, k), we assume k = 1
      params_In(ref_In.layout()),
      ref_In(ref_In),
      params_Window(layout::RowMajor(kInnerConv ? problem_size.n() : problem_size.m()), window_size, channels),
      ref_conv_Window(ref_conv_Window), // do not pass explicit window, we construct it later
      params_C(ref_C.layout()),
      ref_C(ref_C),
      params_D(ref_D.layout()),
      ref_D(ref_D),
      output_op(output_op) {
      gemm_k_size = calc_gemm_k_size(problem_size, grid_tiled_shape);
      // if (threadIdx.x == 0){
        printf("gemm_k_size: %d \n", gemm_k_size);
      // }

      semaphore = workspace;
    }

    CUTLASS_HOST_DEVICE
    static int calc_gemm_k_size(gemm::GemmCoord const &problem_size, GemmCoord const &grid_tiled_shape) {
      // total tiles that cover the k-dim
      int total_gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
      // how many iterations per block in the grid (grid_tiled_shape.k() is assumed to be 1)
      int gemm_k_iterations = (total_gemm_k_iterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();
      printf("total_gemm_k_iterations %d gemm_k_iterations %d\n", total_gemm_k_iterations, gemm_k_iterations);
      return gemm_k_iterations * Mma::Shape::kK;
    }

  };

  using HostParams = std::vector<SampleParams>;
  struct Params {
    int sample_count;
    SampleParams *params;
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
  // TODO(klecki): B (window) alignement
    static Status can_implement(
      cutlass::gemm::GemmCoord const & problem_size,
      typename Mma::IteratorA::TensorRef ref_In,
      // typename Mma::IteratorB::TensorRef ref_B,
      typename Epilogue::OutputTileIterator::TensorRef ref_C,
      typename Epilogue::OutputTileIterator::TensorRef ref_D) {

    static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    if (!TensorRef_aligned(ref_In, kAlignmentA)) {
      return Status::kErrorMisalignedOperand;
    }

    // if (!TensorRef_aligned(ref_B, kAlignmentB)) {
    //   return Status::kErrorMisalignedOperand;
    // }

    if (!TensorRef_aligned(ref_C, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_D, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if ((problem_size.m() % kAlignmentA) || (problem_size.k() % kAlignmentA) ||
      // (problem_size.n() % kAlignmentB) || (problem_size.k() % kAlignmentB) ||
      (problem_size.m() % kAlignmentC) || (problem_size.n() % kAlignmentC)) {

      return Status::kErrorMisalignedOperand;
    }

    return Status::kSuccess;
  }

  CUTLASS_DEVICE
  void transfer_conv_window(Params const &params_vec, typename Mma::TensorRefWindow &window_smem) {
    SampleParams const &params = params_vec.params[0]; // TODO(klecki): temp for compilation

    ////////////
    //  Copy the window from global mem to smem for matrix bulding lookups
    // Load from params.ref_conv_Window to shared_storage.main_loop.operand_Window

    // int const kWindowLength = Mma::SharedStorage::ShapeWindow;
    // The PredicateTileIterator expects PitchLinearShape and PitchLinear layout.
    // The target shape is RowMajor<1, ConvMmaBase::kWindowLength>, we map it to PitchLinear
    // using WindowShape = typename Mma::SharedStorage::ShapeWindow;
    using WindowShape = layout::PitchLinearShape<kThreadCount, 1>;

    // ThreadMaps define how threads are mapped to a given tile. The PitchLinearStripminedThreadMap
    // stripmines a pitch-linear tile among a given number of threads, first along the contiguous
    // dimension then along the strided dimension.
    using WindowThreadMap = transform::PitchLinearStripminedThreadMap<WindowShape, kThreadCount>;

    // Define the PredicateTileIterator, using TileShape, Element, Layout, and ThreadMap types
    using WindowIteratorGmem = transform::threadblock::PredicatedTileIterator<
        WindowShape, WindowElement, WindowLayout, 0, WindowThreadMap>;

    using WindowIteratorSmem = transform::threadblock::PredicatedTileIterator<
        WindowShape, WindowElement, WindowLayout, 0, WindowThreadMap>;

    // cutlass::Coord<2> window_extent = cutlass::make_Coord(params.window_size, 1);

    // int iterations = (params.window_size + WindowShape::kContiguous - 1) / WindowShape::kContiguous;

    // TODO(klecki): faked window that is already preprocessed
    cutlass::Coord<2> window_extent = cutlass::make_Coord(1024, 1);
    int iterations = (1024 + WindowShape::kContiguous - 1) / WindowShape::kContiguous;

    WindowIteratorGmem src_iterator(params.ref_conv_Window.layout(), params.ref_conv_Window.data(), window_extent, threadIdx.x);
    WindowIteratorSmem dst_iterator(window_smem.layout(), window_smem.data(), window_extent, threadIdx.x);

    typename WindowIteratorGmem::Fragment fragment;
    // using Transform = NumericArrayConverter<
    //   WindowElement,
    //   WindowElement,
    //   WindowIteratorGmem::Fragment::kElements>;

    // Transform transform_window = Transform();

    fragment.clear();

    src_iterator.load(fragment);
    // debug::dump_fragment(fragment, 1);
    dst_iterator.store(fragment);
    ++src_iterator;
    ++dst_iterator;

    for(; iterations > 1; --iterations) {
      src_iterator.load(fragment);
      dst_iterator.store(fragment);
      ++src_iterator;
      ++dst_iterator;
    }
    __syncthreads();
  }

   /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params_vec, SharedStorage &shared_storage) {
    SampleParams const &params = params_vec.params[0]; // TODO(klecki): temp for compilation

    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

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

    // the offset to the resulting matrix
    cutlass::MatrixCoord tb_offset_C{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.n() * Mma::Shape::kN
    };

    // effective span of the window in the generated matrix
    int radius_span = (params.window_size / 2) * params.channels;
    int window_span = params.window_size * params.channels;


    // We need to start at aligned tile, otherwise tensor ops aren't happy.
    // Take this into account when calculating the non-zero region
    int k_skipped_offset = max(0, threadblock_tile_offset.n() * Mma::Shape::kN - radius_span);
    k_skipped_offset = (k_skipped_offset & ~(Mma::Shape::kK - 1));
    // k_skipped_offset = 0;

    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      k_skipped_offset // 0 * K_SIZE
    };

    cutlass::MatrixCoord tb_offset_B{
      k_skipped_offset,
      threadblock_tile_offset.n() * Mma::Shape::kN
    };

    // todo(klecki): and here are the tile coordinates, woohoo
    // PRINT_IF
    // if (threadIdx.x == 0)
    //   printf("kernel::Conv::operator() threadIdx: (%d, %d, %d), A  row, col:(%d, %d), B row, col: (%d, %d) \n",
    //   threadIdx.x, threadIdx.y, threadIdx.z, tb_offset_A.row(), tb_offset_A.column(), tb_offset_B.row(), tb_offset_B.column());

    // Problem size is a function of threadblock index in the K dimension
    int problem_size_k = min(
      params.problem_size.k(),
      (threadblock_tile_offset.k() + 1) * params.gemm_k_size);
    // Compute threadblock-scoped matrix multiply-add
    // this is how many iterations we need if we start at the offset
    int gemm_k_iterations = (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;
    // this is how many iterations (from the starting offset) is expected to be non-zero
    int nonzero_k_iterations = min((Mma::Shape::kN + window_span + 2 * Mma::Shape::kK - 1) / Mma::Shape::kK, gemm_k_iterations);
    int end_iteration =  gemm_k_iterations - nonzero_k_iterations;
    // end_iteration = 0;
    // int gemm_k_iterations = (iteration_size_k + Mma::Shape::kK - 1) / Mma::Shape::kK;
    // if (!threadIdx.x)
    //   printf("problem_size_k %d tb_offset_A.column() %d Mma::Shape::kK %d\n", problem_size_k, tb_offset_A.column(), Mma::Shape::kK);

    PRINT_IF
      printf("kernel::Conv::operator() threadIdx: (%d, %d, %d), problem_size_k: %d, gemm_k_iterations: %d \n",
        threadIdx.x, threadIdx.y, threadIdx.z, problem_size_k, gemm_k_iterations);

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    auto window_smem = shared_storage.main_loop.operand_Window_ref(params.window_size);
    // Transfer window from gmem to smem <- this is cheap (klecki)
    transfer_conv_window(params_vec, window_smem);



    // TODO(klecki):
    // either compute the window directly in SMEM or get it from GMEM

    // for (int i = thread_idx; i < params.window_sizes[1]; i += kThreadCount) {
    //   window.data()[i] = params.windows[1][i];
    // }
    // __syncthreads();

    void *in_data = params.ref_In.data();
    void *window_data = window_smem.data();

    // Construct iterators to A and B operands
    // Global mem iterators
    typename Mma::IteratorA iterator_A(
      select_A<kInnerConv>(params.params_In, params.params_Window),
      select_A<kInnerConv>(params.ref_In.data(), window_smem.data()),
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A);


    typename Mma::IteratorB iterator_B(
      // Fake stride
      select_B<kInnerConv>(params.params_In, params.params_Window),
      // Pointer to the smem that would be sampled
      select_B<kInnerConv>(params.ref_In.data(), window_smem.data()),
      {problem_size_k, params.problem_size.n()},
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
      mma(gemm_k_iterations, end_iteration, accumulators, iterator_A, iterator_B, accumulators, tb_offset_C);
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
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
      params.params_D,
      params.ref_D.data(),
      params.problem_size.mn(),
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

