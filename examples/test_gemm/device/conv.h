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
#include "cutlass/device_kernel.h"

#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
// #include "cutlass/gemm/kernel/gemm.h"
#include "kernel/conv.h"

// #include "cutlass/gemm/kernel/default_gemm.h"
#include "kernel/default_conv.h"
#include "device/default_conv_configuration.h"

#include "dali/utility.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

/*! Conv device-level operator. This is an interface to efficient CUTLASS GEMM kernels that may
  be invoked from host code.

  The contributions of this class are:

    1. At compile time, it maps data types and high-level structural parameters onto
       specific CUTLASS components.

    2. At runtime, it maps logical arguments to GEMM problems to kernel parameters.

    3. At runtime, it launches kernels on the device.

  The intent is to provide a convenient mechanism for interacting with most plausible GEMM
  configurations for each supported architecture. Consequently, not all parameters are exposed
  to the top-level interface. Rather, sensible defaults at each level of the CUTLASS hierarchy
  are selected to tradeoff simplicity of the interface with flexibility. We expect
  most configurations to be specified at this level. Applications with more exotic requirements
  may construct their kernels of interest using CUTLASS components at the threadblock, warp,
  and thread levels of abstraction.

  CUTLASS exposes computations using the functor design pattern in which objects compose some
  internal state with an overloaded function call operator. This enables decoupling of
  initialization from execution, possibly reducing overhead during steady state phases of
  application execution.

  CUTLASS device-level operators expose an Arguments structure encompassing each logical
  input to the computation. This is distinct from the kernel-level Params structure pattern
  which contains application-specific precomputed state needed by the device code.

  Example of a CUTLASS GEMM operator implementing the functionality of cuBLAS's SGEMM NN
  is as follows:

    //
    // Instantiate the CUTLASS GEMM operator.
    //

    cutlass::gemm::device::Conv<
      float,
      cutlass::layout::ColumnMajor,
      float,
      cutlass::layout::ColumnMajor,
      float,
      cutlass::layout::ColumnMajor
    > gemm_op;

    //
    // Launch the GEMM operation on the device
    //

    cutlass::Status status = gemm_op({
      {m, n, k},                          // GemmCoord problem_size,
      {A, lda},                           // TensorRef<float, layout::ColumnMajor> ref_In,
      {B, ldb},                           // TensorRef<float, layout::ColumnMajor> ref_Windows,
      {C, ldc},                           // TensorRef<float, layout::ColumnMajor> ref_C,
      {D, ldd},                           // TensorRef<float, layout::ColumnMajor> ref_D,
      {alpha, beta}                       // EpilogueOutputOp::Params epilogue_op_params
    });


  A simplified view of the template is listed below.

    template <
      /// Element type for A matrix operand
      typename ElementIn,

      /// Layout type for A matrix operand
      typename LayoutIn,

      /// Element type for B matrix operand
      typename ElementWindow,

      /// Layout type for B matrix operand
      typename LayoutWindow,

      /// Element type for C and D matrix operands
      typename ElementOut,

      /// Layout type for C and D matrix operands
      typename LayoutOut,

      /// Element type for internal accumulation
      typename ElementAccumulator,

      /// Operator class tag
      typename OperatorClass,

      /// Tag indicating architecture to tune for
      typename ArchTag,

      /// Threadblock-level tile size (concept: GemmShape)
      typename ThreadblockShape,

      /// Warp-level tile size (concept: GemmShape)
      typename WarpShape,

      /// Warp-level tile size (concept: GemmShape)
      typename InstructionShape,

      /// Epilogue output operator
      typename EpilogueOutputOp,

      /// Threadblock-level swizzling operator
      typename ThreadblockSwizzle,

      /// Number of stages used in the pipelined mainloop
      int Stages
    >
    class Conv;
*/
template <
    /// Element type for input matrix operand
    typename ElementIn_,
    /// Element type for input matrix operand
    typename ElementCastIn_,
    /// Layout type for input matrix operand
    typename LayoutIn_,
    /// Element type for window operands
    typename ElementWindow_,
    /// Element type for window operands
    typename ElementCastWindow_,
    /// Element type for C and D matrix operands
    typename ElementOut_,
    /// Layout type for C and D matrix operands
    typename LayoutOut_,
    // Number of input data axes to process
    int Axes = 2,
    bool InnerConv = true,
    /// Element type for internal accumulation
    typename ElementAccumulator_ = ElementOut_,
    /// Operator class tag
    typename OperatorClass_ = arch::OpClassSimt,
    /// Tag indicating architecture to tune for
    typename ArchTag_ = arch::Sm70,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_ = typename DefaultConvConfiguration<OperatorClass_, ArchTag_,
        // TODO(klecki): ElementIn_ or ElementCastIn_?
        select_A_t<InnerConv, ElementIn_, ElementWindow_>,
        select_B_t<InnerConv, ElementIn_, ElementWindow_>,
        ElementOut_, ElementAccumulator_>::ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_ = typename DefaultConvConfiguration<OperatorClass_, ArchTag_,
        select_A_t<InnerConv, ElementIn_, ElementWindow_>,
        select_B_t<InnerConv, ElementIn_, ElementWindow_>,
        ElementOut_, ElementAccumulator_>::WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_ = typename DefaultConvConfiguration<OperatorClass_, ArchTag_,
        select_A_t<InnerConv, ElementIn_, ElementWindow_>,
        select_B_t<InnerConv, ElementIn_, ElementWindow_>,
        ElementOut_, ElementAccumulator_>::InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp_ = typename DefaultConvConfiguration<OperatorClass_, ArchTag_,
        select_A_t<InnerConv, ElementIn_, ElementWindow_>,
        select_B_t<InnerConv, ElementIn_, ElementWindow_>,
        ElementOut_, ElementAccumulator_>::EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_ =
        typename threadblock::GemmIdentityThreadblockSwizzle<>,
    /// Number of stages used in the pipelined mainloop
    int Stages = DefaultConvConfiguration<OperatorClass_, ArchTag_,
        select_A_t<InnerConv, ElementIn_, ElementWindow_>,
        select_B_t<InnerConv, ElementIn_, ElementWindow_>,
        ElementOut_, ElementAccumulator_>::kStages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA = DefaultConvConfiguration<OperatorClass_, ArchTag_,
        select_A_t<InnerConv, ElementIn_, ElementWindow_>,
        select_B_t<InnerConv, ElementIn_, ElementWindow_>,
        ElementOut_, ElementAccumulator_>::kAlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB = DefaultConvConfiguration<OperatorClass_, ArchTag_,
        select_A_t<InnerConv, ElementIn_, ElementWindow_>,
        select_B_t<InnerConv, ElementIn_, ElementWindow_>,
        ElementOut_, ElementAccumulator_>::kAlignmentB,
    /// If true, kernel supports split-K with serial reduction
    bool SplitKSerial = false,
    /// Operation performed by GEMM
    typename Operator_ = typename DefaultConvConfiguration<OperatorClass_, ArchTag_,
        select_A_t<InnerConv, ElementIn_, ElementWindow_>,
        select_B_t<InnerConv, ElementIn_, ElementWindow_>,
        ElementOut_, ElementAccumulator_>::Operator,
    /// Whether Beta is zero or not
    bool IsBetaZero = false>
class Conv {
 public:

  using ElementIn = ElementIn_;
  using ElementCastIn = ElementCastIn_;
  using LayoutIn = LayoutIn_;
  // using TensorRefA = TensorRef<ElementIn const, LayoutIn>;
  using ElementWindow = ElementWindow_;
  using ElementCastWindow = ElementCastWindow_;
  using LayoutWindow = layout::RowMajor; // placeholder
  // using TensorRefB = TensorRef<ElementWindow const, LayoutWindow>;
  using ElementOut = ElementOut_;
  using LayoutOut = LayoutOut_;
  using TensorRefC = TensorRef<ElementOut const, LayoutOut>;
  using TensorRefD = TensorRef<ElementOut, LayoutOut>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Operator = Operator_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static bool const kSplitKSerial = SplitKSerial;
  static bool const kIsBetaZero = IsBetaZero;
  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;
  static_assert(kSplitKSerial == false, "Only basic options are supported");
  static int const split_k_slices = 1; // TODO(klecki): investigate how this can be used, now assume 1
  static int const kAxes = Axes;
  static bool const kInnerConv = InnerConv;

  /// Define the kernel, SIMT
  using ConvKernel = typename kernel::DefaultConv<
    select_A_t<InnerConv, ElementIn_, ElementWindow_>,
    select_A_t<InnerConv, ElementCastIn_, ElementCastWindow_>,
    LayoutIn,
    kAlignmentA,
    select_B_t<InnerConv, ElementIn_, ElementWindow_>,
    select_B_t<InnerConv, ElementCastIn_, ElementCastWindow_>,
    LayoutWindow,
    kAlignmentB,
    ElementOut,
    LayoutOut,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    kStages,
    kSplitKSerial,
    Operator,
    kIsBetaZero,
    kInnerConv
  >::GemmKernel;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //
    // GemmCoord problem_size;
    Array<int, kAxes> matrix_size;
    int window_size;
    int channels;
    TensorRef<ElementIn const, LayoutIn> ref_In;
    ElementWindow* ref_Window;
    TensorRef<ElementOut const, LayoutOut> ref_C;
    TensorRef<ElementOut, LayoutOut> ref_D;
    typename EpilogueOutputOp::Params epilogue;

    //
    // Methods
    //

    /// Default ctor
    // CUTLASS_HOST_DEVICE
    // Arguments(): problem_size(0, 0, 0), split_k_slices(1) {

    // }

    /// Constructs an Arguments structure
    CUTLASS_HOST_DEVICE
    Arguments(
      Array<int, kAxes> matrix_size_,
      int window_size_,
      int channels_,
      TensorRef<ElementIn const, LayoutIn> ref_In_,
      ElementWindow* ref_Window_,
      TensorRef<ElementOut const, LayoutOut> ref_C_,
      TensorRef<ElementOut, LayoutOut> ref_D_,
      typename EpilogueOutputOp::Params epilogue_ =
        typename EpilogueOutputOp::Params()
    ):
      matrix_size(matrix_size_),
      window_size(window_size_),
      channels(channels_),
      ref_In(ref_In_),
      ref_Window(ref_Window_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      epilogue(epilogue_){

    }
  };

private:

  /// Kernel parameters object
  typename ConvKernel::Params params_;

public:

  /// Constructs the GEMM.
  Conv() { }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    // we assume kSplitKSerial == false, so this will prevent split_k_slices != 1
    if (!kSplitKSerial && split_k_slices > 1) {
      return Status::kErrorInvalidProblem;
    }

    //TODO(klecki): fixup
    // Status status = GemmKernel::can_implement(
    //   args.problem_size,
    //   args.ref_In.non_const_ref(),
    //   args.ref_Windows.non_const_ref(),
    //   args.ref_C.non_const_ref(),
    //   args.ref_D
    // );

    // if (status != Status::kSuccess) {
    //   return status;
    // }

    return Status::kSuccess;
  }

  // /// Gets the workspace size
  // static size_t get_workspace_size(Arguments const &args) {

  //   size_t bytes = 0;

  //   // Determine grid shape
  //   ThreadblockSwizzle threadblock_swizzle;

  //   cutlass::gemm::GemmCoord tiled_shape = threadblock_swizzle.get_tiled_shape(
  //     args.problem_size,
  //     {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
  //     split_k_slices);

  //   printf(">> get_workspace_size:\nProblem Size: (%d, %d, %d), block (%d, %d, %d), k_slices: %d -> shape (%d, %d, %d)\n",
  //      args.problem_size.m(), args.problem_size.n(), args.problem_size.k(),
  //      ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK,
  //      split_k_slices,
  //      tiled_shape.m(), tiled_shape.n(), tiled_shape.k());

  //   if (kSplitKSerial && split_k_slices > 1) {

  //     bytes += sizeof(int) * size_t(tiled_shape.m()) * size_t(tiled_shape.n());
  //   }

  //   return bytes;
  // }

  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    // The basic threadblock swizzle takes only M and N dims into account here
    int dummy_k = 1;
    GemmCoord problem_size(args.matrix_size[0], args.matrix_size[1] * args.channels, dummy_k);
    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
      problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      split_k_slices);

    printf(">> initialize:\nProblem Size: (%d, %d, %d), block (%d, %d, %d), k_slices: %d -> grid_shape (%d, %d, %d)\n",
       problem_size.m(), problem_size.n(), problem_size.k(),
       ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK,
       split_k_slices,
       grid_shape.m(), grid_shape.n(), grid_shape.k());

    // if (kSplitKSerial) {
    //   if (split_k_slices > 1) {
    //     if (!workspace) {
    //       return Status::kErrorWorkspaceNull;
    //     }

    //     size_t bytes = get_workspace_size(args);

    //     cudaError_t result = cudaMemsetAsync(workspace, 0, bytes, stream);

    //     if (result != cudaSuccess) {
    //       return Status::kErrorInternal;
    //     }
    //   }
    // }
    // else {

    //   if (split_k_slices > 1) {
    //     return Status::kErrorInvalidProblem;
    //   }
    // }

    // Initialize the Params structure
    params_ = typename ConvKernel::Params{
      args.channels,
      GetProblemSize(args.matrix_size, args.channels, kInnerConv),
      grid_shape,
      args.ref_In.non_const_ref(),
      {args.ref_Window, {args.window_size}}, // build window ref on the fly
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.epilogue,
      static_cast<int *>(workspace)
    };


    return Status::kSuccess;
  }

    // TODO(klecki): this just swaps the pointers, but we actually need to swap the sizes as weel,
    // need to use initialize
  // /// Lightweight update given a subset of arguments
  // Status update(Arguments const &args, void *workspace = nullptr) {

  //   if (kSplitKSerial && args.split_k_slices > 1) {
  //     if (!workspace) {
  //       return Status::kErrorWorkspaceNull;
  //     }
  //   }

  //   params_.ref_In.reset(args.ref_In.non_const_ref().data());
  //   params_.ref_Windows.reset(args.ref_Windows.non_const_ref().data());
  //   params_.ref_C.reset(args.ref_C.non_const_ref().data());
  //   params_.ref_D.reset(args.ref_D.data());
  //   params_.output_op = args.epilogue;
  //   params_.semaphore = static_cast<int *>(workspace);

  //   return Status::kSuccess;
  // }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {

    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(ConvKernel::kThreadCount, 1, 1);

    cudaError_t result;

    int smem_size = int(sizeof(typename ConvKernel::SharedStorage));
    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(Kernel<ConvKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }

      result = cudaFuncSetAttribute(
          Kernel<ConvKernel>,
          cudaFuncAttributePreferredSharedMemoryCarveout, 100);

      if (result != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    cutlass::Kernel<ConvKernel><<<grid, block, smem_size, stream>>>(params_);

    #if 1
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);


    for (int i = 0; i < 100; i++) {
      cutlass::Kernel<ConvKernel><<<grid, block, smem_size, stream>>>(params_);
    }

    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float totalTime;
    cudaEventElapsedTime(&totalTime, start, stop);
    std::cout << "total time " << totalTime / 100.0 << " ms\n";

    #endif

    result = cudaGetLastError();

    return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args,
    void *workspace = nullptr,
    cudaStream_t stream = nullptr) {

    Status status = initialize(args, workspace);

    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }

  GemmCoord GetProblemSize(const Array<int, kAxes> &matrix_size, int channels, bool inner) {
    if (inner) {
      // (m, n, n) where n = width * channels
      return {matrix_size[0], matrix_size[1] * channels, matrix_size[1] * channels};
    } else {
      // m, n, m where n = width * channels
      return {matrix_size[0], matrix_size[1] * channels, matrix_size[0]};
    }
  }
};


} // namespace device
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
