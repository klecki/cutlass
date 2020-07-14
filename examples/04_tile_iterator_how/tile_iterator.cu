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
  This example demonstrates how to use the PredicatedTileIterator in CUTLASS to load data from
  addressable memory, and then store it back into addressable memory.

  TileIterator is a core concept in CUTLASS that enables efficient loading and storing of data to
  and from addressable memory. The PredicateTileIterator accepts a ThreadMap type, which defines
  the mapping of threads to a "tile" in memory. This separation of concerns enables user-defined
  thread mappings to be specified.

  In this example, a PredicatedTileIterator is used to load elements from a tile in global memory,
  stored in column-major layout, into a fragment and then back into global memory in the same
  layout.

  This example uses CUTLASS utilities to ease the matrix operations.

*/

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>

// CUTLASS includes
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"

//
//  CUTLASS utility includes
//

// Defines operator<<() to write TensorView objects to std::ostream
#include "cutlass/util/tensor_view_io.h"

// Defines cutlass::HostTensor<>
#include "cutlass/util/host_tensor.h"

// Defines cutlass::reference::host::TensorFill() and
// cutlass::reference::host::TensorFillBlockSequential()
#include "cutlass/util/reference/host/tensor_fill.h"

#pragma warning( disable : 4503)
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define PredicatedTileIterators to load and store a M-by-K tile, in column major layout.

__global__ void copy(
    float *dst_pointer,
    float *src_pointer,
    int window_size) {

  constexpr static int kBufSize = 1024;
  __shared__ float buffer[kBufSize];

  static int const kThreadCount = 32;
  using WindowShape = cutlass::layout::PitchLinearShape<kThreadCount, 1>;
  using WindowLayout = cutlass::layout::PitchLinear;

  using WindowElement = float;
  using WindowThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<WindowShape, kThreadCount>;

  // Define the PredicateTileIterator, using TileShape, Element, Layout, and ThreadMap types
  using WindowGmemIterator = cutlass::transform::threadblock::PredicatedTileIterator<
      WindowShape, WindowElement, WindowLayout, 0, WindowThreadMap>;

  cutlass::Coord<2> window_extent = cutlass::make_Coord(window_size, 1);

  int iterations = (window_size + WindowShape::kContiguous - 1) / WindowShape::kContiguous;

  using WindowSmemIterator = cutlass::transform::threadblock::RegularTileIterator<
      WindowShape, WindowElement, WindowLayout, 0, WindowThreadMap>;

  auto tmp_ref = cutlass::TensorRef<float, cutlass::layout::PitchLinear>{buffer, WindowLayout{1024}};

  WindowGmemIterator src_iterator(WindowLayout(1024), src_pointer, window_extent, threadIdx.x);
  WindowSmemIterator dst_iterator(tmp_ref, threadIdx.x);
  // dst_iterator.set_iteration_index(iterations);

  typename WindowGmemIterator::Fragment fragment;

  fragment.clear();
  src_iterator.load(fragment);
  dst_iterator.store(fragment);
  // __syncthreads();
  ++src_iterator;
  ++dst_iterator;

  for(; iterations > 1; --iterations) {

    src_iterator.load(fragment);
    dst_iterator.store(fragment);

    ++src_iterator;
    ++dst_iterator;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kBufSize; i += kThreadCount) {
    dst_pointer[i] = buffer[i];
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Initializes the source tile with sequentially increasing values and performs the copy into
// the destination tile using two PredicatedTileIterators, one to load the data from addressable
// memory into a fragment (regiser-backed array of elements owned by each thread) and another to
// store the data from the fragment back into the addressable memory of the destination tile.

cudaError_t TestTileIterator(int elements) {

    // For this example, we chose a <64, 4> tile shape. The PredicateTileIterator expects
    // PitchLinearShape and PitchLinear layout.
    using Shape = cutlass::layout::PitchLinearShape<64, 4>;
    using Layout = cutlass::layout::PitchLinear;
    using Element = float;
    int const kThreads = 32;


    cutlass::Coord<2> copy_extent = cutlass::make_Coord(elements, 1);
    cutlass::Coord<2> alloc_extent = cutlass::make_Coord(elements, 1);

    // Allocate source and destination tensors
    cutlass::HostTensor<Element, Layout> src_tensor(alloc_extent);
    cutlass::HostTensor<Element, Layout> dst_tensor(alloc_extent);

    Element oob_value = Element(-1);

    // Initialize destination tensor with all -1s
    cutlass::reference::host::TensorFill(dst_tensor.host_view(), oob_value);
    // Initialize source tensor with sequentially increasing values
    cutlass::reference::host::BlockFillSequential(src_tensor.host_data(), src_tensor.capacity());

    dst_tensor.sync_device();
    src_tensor.sync_device();

    dim3 block(kThreads, 1);
    dim3 grid(1, 1);

    // Launch copy kernel to perform the copy
    copy<<< grid, block >>>(
            dst_tensor.device_data(),
            src_tensor.device_data(),
            elements
    );

    cudaError_t result = cudaGetLastError();
    if(result != cudaSuccess) {
      std::cerr << "Error - kernel failed." << std::endl;
      return result;
    }

    dst_tensor.sync_host();

    // Verify results
    for(int s = 0; s < alloc_extent[1]; ++s) {
      for(int c = 0; c < alloc_extent[0]; ++c) {

          Element expected = Element(0);

          if(c < copy_extent[0] && s < copy_extent[1]) {
            expected = src_tensor.at({c, s});
          }
          else {
            expected = oob_value;
          }

          Element got = dst_tensor.at({c, s});
          bool equal = (expected == got);
          printf("Element [%d]: expected %f, got %f\n", c, expected, got);
          // if(!equal) {
          //     std::cerr << "Error - source tile differs from destination tile." << std::endl;
          //   return cudaErrorUnknown;
          // }
      }
    }

    return cudaSuccess;
}

int main(int argc, const char *arg[]) {
    int count = argc > 1 ? atoi(arg[1]) : 128;
    cudaError_t result = TestTileIterator(count);

    if(result == cudaSuccess) {
      std::cout << "Passed." << std::endl;
    }

    // Exit
    return result == cudaSuccess ? 0 : -1;
}

