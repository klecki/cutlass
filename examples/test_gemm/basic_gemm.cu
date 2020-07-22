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

/*
  This example demonstrates how to call a CUTLASS GEMM kernel and provides a naive reference
  matrix multiply kernel to verify its correctness.

  The CUTLASS Gemm template is instantiated in the function CutlassSgemmNN. This is kernel computes
  the general matrix product (GEMM) using single-precision floating-point arithmetic and assumes
  all matrices have column-major layout.

  The threadblock tile size is chosen as 128x128x8 which offers good performance for large matrices.
  See the CUTLASS Parallel for All blog post for more exposition on the tunable parameters available
  in CUTLASS.

  https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

  Aside from defining and launching the SGEMM kernel, this example does not use any other components
  or utilities within CUTLASS. Such utilities are demonstrated elsewhere in other examples and are
  prevalent in the CUTLASS unit tests.

  This example has delibrately been kept similar to the basic_gemm example from cutass-1.3 to
  highlight the minimum amount of differences needed to transition to cutlass-2.0.

  Cutlass-1.3 sgemm: https://github.com/NVIDIA/cutlass/blob/master/examples/00_basic_gemm/basic_gemm.cu
*/

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>

#include <dbg.h>

// Helper methods to check for errors
#include "helper.h"

//
// CUTLASS includes needed for single-precision GEMM kernel
//
#include "device/conv.h"
#include "cutlass/array.h"

// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/half.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////


static constexpr int kWindowSize = 17;

using A_type = float;
using B_type = float;
using C_type = float;

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  int window_size,
  A_type alpha,
  A_type const *A,
  int lda,
  B_type const *window,
  C_type beta,
  C_type *C,
  int ldc) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`

  using ColumnMajor = cutlass::layout::ColumnMajor;
  using RowMajor = cutlass::layout::RowMajor;

  // using CutlassConv = cutlass::gemm::device::DaliExtendedGemm<A_type,        // Data-type of A matrix
  //                                                 ColumnMajor,  // Layout of A matrix
  //                                                 B_type,        // Data-type of B matrix
  //                                                 ColumnMajor,  // Layout of B matrix
  //                                                 C_type,        // Data-type of C matrix
  //                                                 ColumnMajor>; // Layout of C matrix


  // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm70;

  // // This code section describes the tile size a thread block will compute
  using ShapeMMAThreadBlock =
      cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
  // // This code section describes tile size a warp will compute
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32
  // // This code section describes the size of MMA op
  // !!!! WE NEED THIS SO IT CAN ACTUALLY RUN ON Tensor Cores, the default is different
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  // <- MMA Op tile M = 8, N = 8, K = 4

  using CutlassConv = cutlass::gemm::device::Conv<A_type,        // Data-type of A matrix
                                                  RowMajor,  // Layout of A matrix
                                                  B_type,        // Data-type of B matrix
                                                  C_type,        // Data-type of C matrix
                                                  RowMajor, 2, false>; // Layout of C matrix


  // using CutlassConv = cutlass::gemm::device::Conv<A_type,        // Data-type of A matrix
  //                                                 RowMajor,  // Layout of A matrix
  //                                                 B_type,        // Data-type of B matrix
  //                                                 C_type,        // Data-type of C matrix
  //                                                 RowMajor,    // Layout of C matrix
  //                                                 2, false, // axes, InnerConv
  //                                                 C_type,  // element acumulator
  //                                                 MMAOp, // tensor op
  //                                                 SmArch, // arch 70
  //                                                 ShapeMMAThreadBlock, // we can probably leave default shapes, but we need gemm 8x8x4
  //                                                 ShapeMMAWarp,
  //                                                 ShapeMMAOp
  //                                                 >;

  // Define a CUTLASS GEMM type
  CutlassConv gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //

  // TODO(klecki): The cutlass::Array doesn't have an aggregate constructor :C
  cutlass::Array<int, 2> size;
  size[0] = M;
  size[1] = N;
  cutlass::Array<int, 2> window_sizes;
  cutlass::Array<B_type *, 2> windows;
  for (int i = 0; i < 2; i++) {
    window_sizes[i] = window_size;
    windows[i] = const_cast<B_type*>(window); // TODO(klecki): passing non-const value, cause CUTLASS is using non-const refs due to RW iterators (even when only reading)
  }
  CutlassConv::Arguments args(size,  // Input matrix dimensions
                              window_sizes, // Window sizes
                              1, // channels count (innermost)
                              {A, lda},    // Tensor-ref for source matrix A
                              windows,    // Pointers to windows
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue

  //
  // Launch the CUTLASS GEMM kernel.
  //

  cutlass::Status status = gemm_operator(args);

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// The source code after this point in the file is generic CUDA using the CUDA Runtime API
// and simple CUDA kernels to initialize matrices and compute the general matrix product.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a matrix with small integers.
template <typename T>
__global__ void InitializeMatrix_kernel(
  T *matrix,
  int ldm,
  int rows,
  int columns,
  int seed = 0) {

  int row = threadIdx.x + blockIdx.x * blockDim.x;
  int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < rows && col < columns) {
    int offset = row * ldm + col;

    // Generate arbitrary elements.
    // int const k = 16807;
    // int const m = 16;
    // T value = T(((offset + seed) * k % m) - m / 2); // TODO modulo something
    // T value = row * 100 + col;

    T value = static_cast<T>(0.f);
    if (row == col)
      value =  static_cast<T>(1.f);

    matrix[offset] = value;

  }
}

template <typename T>
__global__ void InitializeMatrix_kernel_col_invariant(
  T *matrix,
  int ldm,
  int rows,
  int columns,
  int seed = 0) {

  int row = threadIdx.x + blockIdx.x * blockDim.x;
  int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < rows && col < columns) {
    int offset = row * ldm + col;

    // Generate arbitrary elements.
    // int const k = 16807;
    // int const m = 16;
    // T value = T(((col + seed) * k % m) - m / 2); // TODO modulo something
    int diag_dist = row - col;
    T value =  static_cast<T>(0.f);
    // int window_size = kWindowSize;
    // int radius = window_size / 2;
    // if (diag_dist == 0) {
    //   value = 100;
    // }
    // else if (::abs(diag_dist) <= radius) {
    //   value = radius - abs(diag_dist);
    // }

    // initialize with the col-repeat scheme
    // T value = col;

    matrix[offset] = value;
  }
}


// // the "B" matrix basically
// template <typename T>
// __global__ void InitializeMatrix_kernel_col_invariant(
//   T *matrix,
//   int ldm,
//   int rows,
//   int columns,
//   int seed = 0) {

//   int row = threadIdx.x + blockIdx.x * blockDim.x;
//   int col = threadIdx.y + blockIdx.y * blockDim.y;

//   if (row < rows && col < columns) {
//     int offset = row * ldm + col;

//     // Generate arbitrary elements.
//     // int const k = 16807;
//     // int const m = 16;
//     // T value = T(((row + seed) * k % m) - m / 2); // TODO modulo something
//     T value =  col;

//     matrix[offset] = value;
//   }
// }

/// Simple function to initialize a matrix to arbitrary small integers.
template <typename T>
cudaError_t InitializeMatrix(T *matrix, int ldm, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel<<< grid, block >>>(matrix, ldm, rows, columns, seed);

  return cudaGetLastError();
}

template <typename T>
cudaError_t InitializeMatrixColInvariant(T *matrix, int ldm, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel_col_invariant<<< grid, block >>>(matrix, ldm, rows, columns, seed);

  return cudaGetLastError();
}
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates device memory for a matrix then fills with arbitrary small integers.
template <typename T>
cudaError_t AllocateMatrix(T **matrix, int ldm, int rows, int columns, int seed = 0, bool col_invariant = false) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(T) * ldm * rows;

  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Initialize matrix elements to arbitrary small integers.
  if (col_invariant) {
    result = InitializeMatrixColInvariant(*matrix, ldm, rows, columns, seed);
  }
  else {
    result = InitializeMatrix(*matrix, ldm, rows, columns, seed);
  }



  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference GEMM computation.
__global__ void ReferenceGemm_kernel(
  int M,
  int N,
  int K,
  float alpha,
  A_type const *A,
  int lda,
  B_type const *B,
  int ldb,
  float beta,
  C_type *C,
  int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    C_type accumulator = 0;

    for (int k = 0; k < K; ++k) {
      accumulator += (float)A[i * lda + k] * (float)B[k * ldb + j];
    }

    C[i * ldc + j] = alpha * accumulator + beta * C[i * ldc + j];
  }
}

/// Reference GEMM computation.
cudaError_t ReferenceGemm(
  int M,
  int N,
  int K,
  float alpha,
  A_type const *A,
  int lda,
  B_type const *B,
  int ldb,
  float beta,
  C_type *C,
  int ldc) {

  dim3 block(16, 16);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );

  ReferenceGemm_kernel<<< grid, block >>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////


// from DALI:
template <typename T>
__host__ __device__
std::enable_if_t<std::is_integral<T>::value, T> idx_reflect_101(T idx, T lo, T hi) {
  if (hi - lo < 2)
    return hi - 1;  // make it obviously wrong if hi <= lo
  for (;;) {
    if (idx < lo)
      idx = 2 * lo - idx;
    else if (idx >= hi)
      idx = 2 * hi - 2 - idx;
    else
      break;
  }
  return idx;
}

/// @brief Equivalent to `idx_reflect_101(idx, 0, size)`
template <typename T>
__host__ __device__
std::enable_if_t<std::is_integral<T>::value, T> idx_reflect_101(T idx, T size) {
  return idx_reflect_101(idx, T(0), size);
}

/// Naive reference Conv computation.
__global__ void ReferenceConv_kernel_inner(
  int M, // rows
  int N, // cols
  int radius,
  float alpha,
  A_type const *A,
  int lda,
  B_type const *window,
  float beta,
  C_type *C,
  int ldc) {

  int row = threadIdx.x + blockIdx.x * blockDim.x;
  int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < M && col < N) {
    C_type accumulator = 0;

    for (int k = -radius; k <= radius; ++k) {
      accumulator += (float)A[row * lda + idx_reflect_101(col + k, N)] * (float)window[k];
    }

    C[row * ldc + col] = alpha * accumulator + beta * C[row * ldc + col];
  }
}


/// Naive reference Conv computation.
__global__ void ReferenceConv_kernel_outer(
  int M, // rows
  int N, // cols
  int radius,
  float alpha,
  A_type const *A,
  int lda,
  B_type const *window,
  float beta,
  C_type *C,
  int ldc) {

  int row = threadIdx.x + blockIdx.x * blockDim.x;
  int col = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < M && col < N) {
    C_type accumulator = 0;

    for (int k = -radius; k <= radius; ++k) {
      accumulator += (float)A[idx_reflect_101(row + k, M) * lda + col] * (float)window[k];
    }

    C[row * ldc + col] = alpha * accumulator + beta * C[row * ldc + col];
  }
}


/// Reference Conv computation.
cudaError_t ReferenceConv(
  int M,
  int N,
  int radius,
  float alpha,
  A_type const *A,
  int lda,
  B_type const *window,
  float beta,
  C_type *C,
  int ldc, bool inner) {

  dim3 block(16, 16);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );
  if (inner)
    ReferenceConv_kernel_inner<<< grid, block >>>(M, N, radius, alpha, A, lda, window, beta, C, ldc);
  else
    ReferenceConv_kernel_outer<<< grid, block >>>(M, N, radius, alpha, A, lda, window, beta, C, ldc);


  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////


template <typename T>
void print_mat(int rows, int cols, const std::vector<T> &mat, int max_rows = -1, int max_cols = -1) {
  if (max_rows == -1)
    max_rows = rows;
  if (max_cols == -1)
    max_cols = cols;
  for (int r = 0; r < max_rows; r++) {
    std::cout << "{";
    for (int c = 0; c < max_cols; c++) {
      std::cout << (float)mat[r * cols + c] << ", ";
    }
    std::cout << "}\n";
  }
  std::cout << std::endl;
}

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
cudaError_t TestCutlassConv(int M, int N, int K, A_type alpha, C_type beta) {
  cudaError_t result;


  int window_size = kWindowSize;
  int radius = window_size / 2;

  //
  // Define several matrices to be used as operands to GEMM kernels.
  //

  // Compute leading dimensions for each matrix.
  int lda = K;
  int ldb = N;
  int ldc = N;

  // Compute size in bytes of the C matrix.
  size_t sizeof_A = sizeof(A_type) * lda * M;
  size_t sizeof_B = sizeof(B_type) * ldb * K;
  size_t sizeof_C = sizeof(C_type) * ldc * M;

  // Define pointers to matrices in GPU device memory.
  A_type *A;
  B_type *B;
  C_type *C_cutlass;
  C_type *C_reference;

  //
  // Allocate matrices in GPU device memory with arbitrary seeds.
  //

  result = AllocateMatrix(&A, lda, M, K, 0);

  B_type *window;
  int max_window = 1024;
  result = cudaMalloc(reinterpret_cast<void **>(&window), sizeof(B_type) * max_window);
  std::vector<B_type> window_host(max_window);
  for (int i = 0; i < radius; i++) {
    window_host[i] =  static_cast<B_type>(i);
    window_host[window_size - 1 - i] =  static_cast<B_type>(i);
  }
  window_host[radius] =  static_cast<B_type>(100);
  // for (int i = 0; i < window_size; i++) {
  //   printf("Window[%d] = %f\n", i, window_host[i]);
  // }
  for (int i = window_size; i < max_window; i++) {
    window_host[i] =  static_cast<B_type>(-42.f);
  }

  result = cudaMemcpy(window, window_host.data(), sizeof(B_type) * max_window, cudaMemcpyHostToDevice);




  if (result !=  cudaSuccess) {
    return result;
  }

  result = AllocateMatrix(&B, ldb, K, N, 17, true);

  if (result !=  cudaSuccess) {
    cudaFree(A);
    return result;
  }

  result = AllocateMatrix(&C_cutlass, ldc, M, N, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    return result;
  }

  result = AllocateMatrix(&C_reference, ldc, M, N, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    cudaFree(C_cutlass);
    return result;
  }

  result = cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  //
  // Launch CUTLASS GEMM.
  //

  result = CutlassSgemmNN(M, N, K, window_size, alpha, A, lda, window, beta, C_cutlass, ldc);

  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  //
  // Verify.
  //

  // Launch reference GEMM
  // result = ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);
  result = ReferenceConv(M, N, radius, alpha, A, lda, window + radius, beta, C_reference, ldc, false);

  if (result != cudaSuccess) {
    std::cerr << "Reference GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  // Copy to host and verify equivalence.
  std::vector<A_type> host_a(lda * M);
  std::vector<B_type> host_b(ldb * K);
  std::vector<C_type> host_cutlass(ldc * M);
  std::vector<C_type> host_reference(ldc * M);

  result = cudaMemcpy(host_a.data(), A, sizeof_A, cudaMemcpyDeviceToHost);
  result = cudaMemcpy(host_b.data(), B, sizeof_B, cudaMemcpyDeviceToHost);
  result = cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy CUTLASS GEMM results: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  result = cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy Reference GEMM results: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  //
  // Free device memory allocations.
  //

  cudaFree(C_reference);
  cudaFree(C_cutlass);
  cudaFree(B);
  cudaFree(A);

  //
  // Test for bit equivalence of results.
  //

  // dbg(host_reference);
  // dbg(host_cutlass);
  std::cout << "CUTLASS A" << std::endl;
  print_mat(M, K, host_a);
  std::cout << "CUTLASS B" << std::endl;
  print_mat(K, N, host_b);

  std::cout << "CUTLASS reference" << std::endl;
  print_mat(M, N, host_reference);

  std::cout << "CUTLASS C" << std::endl;
  print_mat(M, N, host_cutlass, M, N);
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      if (host_cutlass[row * ldc + col] != host_reference[row * ldc + col]) {
        std::cerr << "CUTLASS results incorrect: (" << row << ", " << col << "): "
          << host_cutlass[row * ldc + col] << " != " << host_reference[row * ldc + col] << std::endl;
          return cudaErrorUnknown;
      }
    }
  }
  if (host_cutlass != host_reference) {
    std::cerr << "CUTLASS results incorrect." << std::endl;

    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to basic_gemm example.
//
// usage:
//
//   00_basic_gemm <M> <N> <K> <alpha> <beta>
//
int main(int argc, const char *arg[]) {

  //
  // Parse the command line to obtain GEMM dimensions and scalar values.
  //

  // GEMM problem dimensions.
  int problem[3] = { 128, 128, 128 };

  for (int i = 1; i < argc && i < 4; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }

  // Scalars used for linear scaling the result of the matrix product.
  A_type scalars[2] = {  static_cast<A_type>(1.f), static_cast<A_type>(0.f) }; //todo scalars assumed to have same type

  // for (int i = 4; i < argc && i < 6; ++i) {
  //   std::stringstream ss(arg[i]);
  //   ss >> scalars[i - 4];
  // }

  //
  // Run the CUTLASS GEMM test.
  //

  cudaError_t result = TestCutlassConv(
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    scalars[0],     // alpha
    scalars[1]      // beta
  );

  if (result == cudaSuccess) {
    std::cout << "Passed." << std::endl;
  }

  // Exit.
  return result == cudaSuccess ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
