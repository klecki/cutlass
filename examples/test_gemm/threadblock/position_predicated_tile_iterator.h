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
    \brief Templates implementing loading of tiles from pitch-linear rank=2 tensors.

    This iterator uses masks to guard out-of-bounds accesses and visits the last "residue" tile
    first, with the objective of minimizing predicate mask updates during steady-state operation.

    A precomputed "Params" object minimizes the amount of state that must be stored in registers,
    and integer addition is used to advance the pointer through memory.
*/

#pragma once

#include "cutlass/arch/memory.h"
#include "threadblock/position_predicated_tile_access_iterator.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////
///
/// Generate Convolution Matrix operand on the fly.
/// We always generate it pretending to be Row-Major, when we get redirected to Pitch-linear
/// implementation, the innermost convolution will be right-hand operand (AdvanceRank = 1),
/// (we iterate over strided-rows), and for outer-convolution the AdvanceRank = 0 (we iterate over
/// contiguous column dim).


/// PositionPredicatedTileIterator
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
/// Regular tile iterator using a precomputed control structure to minimize register liveness
/// and integer arithmetic.
///
/// Layout is assumed to be invariant at the time the precomputed "Params" object is constructed.
///
/// Base pointer and tensor extents may be specified at the time the iterator is constructed.
/// Subsequently, they are assumed to be immutable.
///
/// Adding a logical coordinate offset may be performed at the time the iterator is constructed.
/// Subsequent additions to logical coordinate offset may be performed but are relatively expensive.
///
/// Visitation order is intended to first visit a "residual" tile that may be partially full in
/// both the advance dimension and the steady-state dimension. This is assumed to be the last
/// tile in the iteration sequence. Advancing an iterator that has just been constructed moves to
/// the first tile that is full in the advance dimension and recomputes predicates. Subsequent
/// accesses may be performed without updating internal predicates and are efficient in terms of
/// live register state and pointer arithmetic instructions.
///
/// To be efficient, this assumes the iterator will be dereferenced and advanced at least once
/// outside any looping structure to minimize integer arithmetic.
///
/// Acceses out of bounds are safe so long as `clear_mask()` is called prior to dereferencing
/// the iterator.
///
///
/// Example:
///
/// An efficient pipeline structure may be constructed as follows:
///
// template <typename Iterator>
// __global__ void kernel(
//   typename Iterator::Params params,
//   typename Iterator::Element *ptr,
//   TensorCoord extent) {
//
//   typename Iterator::Fragment fragment;
//
//   TensorCoord threadblock_offset(0, 0);
//
//   Iterator iter(params, ptr, extent, threadIdx.x, threadblock_offsets);
//
//
//   fragment = *iter;        // load "residue" tile first
//   ++iter;                  // advance to first "steady state" tile and update internal masks
//
//
//   #pragma unroll
//   for (int i = Remaining - 1; i >= 0; --i) {
//
//     f(fragment);
//
//     if (!i) {
//       iter.clear_mask();   // light-weight operation to clear masks - subsequent loads become NO-OPs.
//     }
//
//     fragment = *iter;      // load tile during "steady state" phase
//     ++iter;                // advance to next tile - lightweight due to steady-state masks
//   }
// }
//
// void host(TensorView<Element, 2, layout::PitchLinear> view) {
//
//   using Iterator = transform::threadblock::PositionPredicatedTileIterator;
//
//   typename Iterator::Params params(view.layout());
//
//   kernel<Iterator>(params, view.data());
// }
//
//
// This iterator is responsible for generating matrix used for computing convolution
// of inner or outer axis with provided kernel window. Matrix is generated on the fly
// from a window stored in shared memory (should also work with global memory).
// Let's call this a "kernel matrix".
//
// N.B. matrix used to apply convolution can be generated ahead of time by doing convolution
// with identity matrix.
//
// For inner convolution, we need to handle channels. For simplicity, first let's consider
// case where channels = 1, let's take window_size = 5 and disregard the border handling.
// We want to convolve gray image, HWC along 'W' axis. For this we multiply the image
// from the right by following matrix, placing the convolution window in columns.
// (Note that the window center is always on the diagonal of the matrix):
//
//  0, -1, -2,  x,  x,  x,  x,  x,
//  1,  0, -1, -2,  x,  x,  x,  x,
//  2,  1,  0, -1, -2,  x,  x,  x,
//  x,  2,  1,  0, -1, -2,  x,  x,
//  x,  x,  2,  1,  0, -1, -2,  x,
//  x,  x,  x,  2,  1,  0, -1, -2,
//  x,  x,  x,  x,  2,  1,  0, -1,
//  x,  x,  x,  x,  x,  2,  1,  0,
//
// where number stands for particular kernel window element, and x for actual 0 value
// Now, let's consider border handling as "repeat 101". If we extended both image and
// the generated matrix by the border elements, we would get:
//
// -2,  x,  x,  x,  x,  x,  x,  x   // -2
// -1, -2,  x,  x,  x,  x,  x,  x   // -1
//  ------------------------------
//  0, -1, -2,  x,  x,  x,  x,  x,  //  0
//  1,  0, -1, -2,  x,  x,  x,  x,  //  1
//  2,  1,  0, -1, -2,  x,  x,  x,  //  2
//  x,  2,  1,  0, -1, -2,  x,  x,
//  x,  x,  2,  1,  0, -1, -2,  x,
//  x,  x,  x,  2,  1,  0, -1, -2,
//  x,  x,  x,  x,  2,  1,  0, -1,
//  x,  x,  x,  x,  x,  2,  1,  0,
//  ------------------------------
//  x,  x,  x,  x,  x,  x,  2,  1,
//  x,  x,  x,  x,  x,  x,  x,  2,
//
// Row "-2" of "kernel matrix" would multiply column "-2" of the image, which is equal to column "2"
// of the image. By the distributive properties of multiplication. We can add the coeficients
// of row -2 and 2 to get the same result without the need to extend the matrices,
// thus the matrix will contain a sum of given kernel window elements in given positions:
//
//  0,    -1,    -2,     x,     x,     x,     x,    x,
//  1 -1,  0 -2, -1,    -2,     x,     x,     x,    x,
//  2 -2,  1,     0,    -1,    -2,     x,     x,    x,
//  x,     2,     1,     0,    -1,    -2,     x,    x,
//  x,     x,     2,     1,     0,    -1,    -2,    x,
//  x,     x,     x,     2,     1,     0,    -1,   -2 2,
//  x,     x,     x,     x,     2,     1,     0 2, -1 1,
//  x,     x,     x,     x,     x,     2,     1,    0,
//
// To generate specific coordinate (row, col), we check how far it is from diagonal:
// diag_dist = row - col <- negative means we're above, that number maps to the base window element
// we need to use.
// When handling the borders, we need to check the distance to the top and bottom of the matrix.
// For the "top" border, we would use elements that are twice the distance to the top from
// currently loaded "base element", than those that are twice the distance to the bottom,
// and repeat the process until we go outside of the window radius. Similarly for the "bottom"
// border.
//
// Examples:
// (1, 0): diag_dist = 1   -> we load the kernel window at coordinate "1".
// The distance to the top is equal 1 (row coordinate), so moving twice the distance up
// gives up 1 - 2 * 1 = -1 -> we need to add the kernel window at coordinate "-1".
// Distance to the bottom is 6, and -1 - 2 * 6 = -13 is out of our kernel window size so we stop.
// Same when going to the bottom, 1 + 2 * 6 = 13 - also out.
// Note that, when the distance is 0, we don't add the additional coefficients in this border mode.
//
// Make channel number > 1 will cause additional gaps in the kernel window when looking at the
// columns of "kernel matrix" for example (again without border) for channels = 3,
// part of the matrix:
//
//  0,  x,  x, -1,  x,  x, -2
//  x,  0,  x,  x, -1,  x,  x
//  x,  x,  0,  x,  x, -1,  x
//  1,  x,  x,  0,  x,  x, -1
//  x,  1,  x,  x,  0,  x,  x
//  x,  x,  1,  x,  x,  0,  x
//  2,  x,  x,  1,  x,  x,  0
//  x,  2,  x,  x,  1,  x,  x
//  x,  x,  2,  x,  x,  1,  x,
//  x,  x,  x,  2,  x,  x,  1,
//  x,  x,  x,  x,  2,  x,  x,
//  x,  x,  x,  x,  x,  2,  x,
//  x,  x,  x,  x,  x,  x,  2
//
// We need to consider a case, when we're using aligned vector loads through an AlignedArray.
// Same principles will apply, but we can observe that the window element indexes decreese
// along the rows. We place the kernel window in reverse in memory (with the necessary empty
// spaces for channels) so we can utilize the vector loads:
//
// -2, x, x, -1, x, x, 0, x, x, 1, x, x, 2
//
// In case of outer convolution, the "kernel matrix" is placed on the left
// and we can ignore the channels - it's the same matrix as for inner convolution but transposed,
// but when used with aligned vector loads it requires a bit different masking (instead of
// ignoring first and last row, we need to mask the first and last column).
// The kernel window is placed in the rows, again with the centers on the diagonal.
//
//  0,  1,  2,  x,  x,  x,  x,  x,
// -1,  0,  1,  2,  x,  x,  x,  x,
// -2, -1,  0,  1,  2,  x,  x,  x,
//  x, -2, -1,  0,  1,  2,  x,  x,
//  x,  x, -2, -1,  0,  1,  2,  x,
//  x,  x,  x, -2, -1,  0,  1,  2,
//  x,  x,  x,  x, -2, -1,  0,  1,
//  x,  x,  x,  x,  x, -2, -1,  0,
//
// Additionally when using vector loads with border handling for the outer kernel matrix,
// we need in turns mirrored window and regular window order.
//
///
///
template <
  typename Shape,
  typename Element,
  typename Layout,
  int AdvanceRank,
  typename ThreadMap,
  int AccessSize = ThreadMap::kElementsPerAccess
>
class PositionPredicatedTileIterator;

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PositionPredicatedTileIterator for pitch-linear data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int AccessSize>
class PositionPredicatedTileIterator<Shape_, Element_, layout::PitchLinear, AdvanceRank,
                             ThreadMap_, AccessSize> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::PitchLinear;
  //
  static int const kAdvanceRank = AdvanceRank;

  // Advance over strided rows (we always use row-major) means right hand side operand
  static int const kInnerConv = kAdvanceRank == 1;



  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  /// Type used for internal memory accesses
  using AccessType = AlignedArray<Element, AccessSize, (AccessSize * sizeof_bits<Element>::value / 8)>;

  /// Underlying iterator to compute the addresses
  using TileAccessIterator =
      PositionPredicatedTileAccessIterator<Shape, Element, Layout, kAdvanceRank,
                                   ThreadMap, AccessType>;

  static int const kAccessesPerVector = TileAccessIterator::kAccessesPerVector;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                               ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename TileAccessIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   public:
    friend PositionPredicatedTileIterator;
    int window_size_;
    int channels_;

   private:
    /// Parameters object
    typename TileAccessIterator::Params params_;

   public:
    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout, int window_size, int channels) :
        params_(layout), window_size_(window_size), channels_(channels) { }

    CUTLASS_HOST_DEVICE
    Params() { }
  };

 private:
  /// Internal pointer type permits fast address arithmetic
  using BytePointer = char *;

 private:
  //
  // Data members
  //

  /// Data member to the tile access iterator
  TileAccessIterator address_iterator_;
  Pointer pointer_;
  int window_size_;
  int channels_ = 1;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to the start of the Window
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : address_iterator_(params.params_, pointer, extent, thread_id,
                          threadblock_offset), pointer_(pointer), window_size_(params.window_size_),
                          channels_(params.channels_) {}

  /// Construct a PositionPredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator(
      Params const &params,  ///< Precomputed parameters object
      Pointer pointer,       ///< Pointer to start of tensor
      TensorCoord extent,    ///< Extent of tensor
      int thread_id          ///< ID of each participating thread
      )
      : PositionPredicatedTileIterator(params, pointer, extent, thread_id,
                               make_Coord(0, 0), 0) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    address_iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator &operator++() {
    if (kAdvanceRank)
      address_iterator_.add_tile_offset({0, 1});
    else
      address_iterator_.add_tile_offset({1, 0});

    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator operator++(int) {
    PositionPredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask() { address_iterator_.clear_mask(); }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() { address_iterator_.enable_mask(); }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { address_iterator_.set_mask(mask); }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { address_iterator_.get_mask(mask); }

  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    load_with_byte_offset(frag, pointer_offset * sizeof_bits<Element>::value / 8);
  }

  // todo(klecki): place in private
  CUTLASS_DEVICE
  void mux(AccessType &dst, const AccessType &lo, const AccessType &hi, int offset) {
    // offset is limited to 0..AccessSize
    offset = ::max(0, ::min(offset, AccessSize));
    #pragma unroll
    for (int i = 0; i < AccessSize - offset; i++) {
      dst[i] = static_cast<Element>(lo[i + offset]);
    }
    #pragma unroll
    for (int i = AccessSize - offset; i < AccessSize; i++) {
      dst[i] = static_cast<Element>(hi[i - AccessSize + offset]);
    }
  }

  CUTLASS_DEVICE
  void mux_add(AccessType &dst, const AccessType &lo, const AccessType &hi, int offset) {
    // offset is limited to 0..AccessSize
    offset = ::max(0, ::min(offset, AccessSize));
    #pragma unroll
    for (int i = 0; i < AccessSize - offset; i++) {
      // TODO(klecki) there is an issue with half, that prohibits me from doing:
      // dst[i] += static_cast<Element>(lo[i + offset]);
      Element tmp = static_cast<Element>(dst[i]) + static_cast<Element>(lo[i + offset]);
      dst[i] = tmp;
    }
    #pragma unroll
    for (int i = AccessSize - offset; i < AccessSize; i++) {
      Element tmp = static_cast<Element>(dst[i]) + static_cast<Element>(hi[i - AccessSize + offset]);
      dst[i] = tmp;
    }
  }

  template <bool mirrored>
  CUTLASS_DEVICE
  int get_lo_offset(int window_element) {
    // for Inner Convolution the window is in reverse when traversing in contiguous axis
    if (mirrored) {
     window_element *= -1;
    }
    if (window_element >= 0) {
      return (window_element / AccessSize) * AccessSize;
    }
    return ((window_element - (AccessSize - 1)) / AccessSize) * AccessSize;
  }

  template <bool mirrored>
  CUTLASS_DEVICE
  int get_offset(int window_element, int lo_offset) {
    if (mirrored) {
      assert(lo_offset <= -window_element);
      return -window_element - lo_offset;
    }
    assert(lo_offset <= window_element);
    return window_element - lo_offset;
  }

  template <bool init, bool mirrored = true>
  CUTLASS_DEVICE
  void load_vec(AccessType &dst, int window_element) {
    int lo_offset = get_lo_offset<mirrored>(window_element);
    int offset = get_offset<mirrored>(window_element, lo_offset);
    constexpr int window_center = kInnerConv || !mirrored ? 256 : 512;
    const auto *access_ptr = reinterpret_cast<AccessType const *>(pointer_ + window_center + lo_offset);
    if (init) {
      mux(dst, access_ptr[0], access_ptr[1], offset);
    } else {
      mux_add(dst, access_ptr[0], access_ptr[1], offset);
    }
  }

  template <bool init, bool mirrored>
  CUTLASS_DEVICE
  void load_vec(AccessType &dst, int window_element, bool mask_first, bool mask_last) {
    int lo_offset = get_lo_offset<mirrored>(window_element);
    int offset = get_offset<mirrored>(window_element, lo_offset);
    constexpr int window_center = kInnerConv || !mirrored ? 256 : 512;
    const auto *access_ptr = reinterpret_cast<AccessType const *>(pointer_ + window_center + lo_offset);
    if (init) {
      mux(dst, access_ptr[0], access_ptr[1], offset);
    } else {
      Element tmp = static_cast<Element>(0);

      if (mask_first) {
        tmp = static_cast<Element>(dst[0]);
      } else if (mask_last) {
        tmp = static_cast<Element>(dst[AccessSize - 1]);
      }

      mux_add(dst, access_ptr[0], access_ptr[1], offset);

      if (mask_first) {
        dst[0] = static_cast<Element>(tmp);
      } else if (mask_last) {
        dst[AccessSize - 1] = static_cast<Element>(tmp);
      }
    }
  }

  CUTLASS_DEVICE
  int Channels() {
    if (kInnerConv) {
      return channels_;
    }
    return 1;
  }

  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, LongIndex byte_offset) {
    PRINT_IF
      printf("LOAD: kStrided: %d, kContiguous: %d, kAccessesPerVector: %d\n", ThreadMap::Iterations::kStrided, ThreadMap::Iterations::kContiguous, kAccessesPerVector);

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {

          int idx = (v + kAccessesPerVector * (c + s * ThreadMap::Iterations::kContiguous));
          // This is to mark the iteration number as a compile time constant
          address_iterator_.set_iteration_index(idx);

          // This calculates the logical coordinate of the beggining of access
          TensorCoord current_coord = address_iterator_.get_current_coord();
          // if (threadIdx.x == 0) {
          //   printf("LOADING COORD: %d %d\n", current_coord.strided(), current_coord.contiguous());
          // }

          // calculate based on the coord and access the pointer_ + appropriate offset
          int major_coord = 0;
          int minor_coord = 0;
          int major_extent = 0;
          if (kInnerConv) {
            // we generate matrix by placing windows vertically, so the row is the major coordinate
            major_coord = current_coord.strided(); // row
            minor_coord = current_coord.contiguous(); // col
            major_extent = address_iterator_.get_extent().strided();
          } else {
            // we generate matrix by placing windows horizontally and there is no channel-based spacing
            major_coord = current_coord.contiguous(); // col
            minor_coord = current_coord.strided(); // row
            major_extent = address_iterator_.get_extent().contiguous();
          }
          // for lhs operand, the problem is transposed and the channel computation can be skipped
          // distance from diagonal in major coordinate direction
          int diag_dist = major_coord - minor_coord; // distance from diagonal - coordinate (x, x), negative are above
          // this is the starting element of the window, for inner-conv vector load goes in negative order
          int window_element = diag_dist;
          int radius = (window_size_ / 2) * Channels();
          // element is used if it's our channel (we're multiple of `channels_` from diagonal)
          // and we still fit in the window
          // TODO(klecki): is_used needs to be checked as if-any in AccessSize
          int farthest_window = window_element < 0 ? window_element + AccessSize - 1 : window_element - AccessSize + 1;
          bool is_used = (::abs(farthest_window) <= radius) && address_iterator_.valid();
              // (kInnerConv ? (window_element == diag_dist) : true) && address_iterator_.valid();
          if (is_used) {
            // Deal with alignment issues
            AccessType dst;
            load_vec<true, kInnerConv>(dst, window_element);

            // Border handling, eliminate the remainder (channel offset from the position calculation,
            // so it is not repeated by every addition - effectivelly we would change the channel)
            int dist_up = (major_coord / Channels()) * Channels();
            int dist_down = ((major_extent - 1 - major_coord) / Channels()) * Channels();
            // add all negative coordinates, pattern is twice the dist up, twice the dist down.
            // we need to have those checks for ANY element in the range
            // TODO(klecki): for now only for the InnerConv - contiguous-coord decreasing
            // access covers window_element .. window_element - AccessSize + 1
            // TODO(klecki): The outer conv needs to mask out some elements of the vector
            // when we are at the edge, and not the whole vectors
            int neg_element = window_element;
            // int neg_compensation = kInnerConv ? 0 : AccessSize - 1;
            int pos_compensation = kInnerConv ? -AccessSize + 1 : -AccessSize + 1;
            int neg_compensation = kInnerConv ? 0 : AccessSize - 1;
            while (true) {
              neg_element -= 2 * dist_up;
              if (-neg_element - neg_compensation <= radius) {
                if (kInnerConv && dist_up >= Channels())
                  load_vec<false>(dst, neg_element);
                else
                  load_vec<false, true>(dst, neg_element, dist_up == 0, false);
              } else {
                break;
              }
              neg_element -= 2 * dist_down;
              if (-neg_element - neg_compensation <= radius) {
                if (kInnerConv && dist_down >= Channels())
                  load_vec<false>(dst, neg_element);
                else
                  load_vec<false, kInnerConv>(dst, neg_element, false, dist_down == AccessSize - 1);
              } else {
                break;
              }
            }
            // add all positive coordinates
            int pos_element = window_element;
            // twice the dist down, twice the dist up
            while (true) {
              pos_element += 2 * dist_down;
              if (pos_element + pos_compensation <= radius) {
                if (kInnerConv && dist_down >= Channels())
                  load_vec<false>(dst, pos_element);
                else
                  load_vec<false, true>(dst, pos_element, false, dist_down == AccessSize - 1);
              } else {
                break;
              }
              pos_element += 2 * dist_up;
              if (pos_element + pos_compensation <= radius) {
                if (kInnerConv && dist_up >= Channels())
                  load_vec<false>(dst, pos_element);
                else
                  load_vec<false, kInnerConv>(dst, pos_element, dist_up == 0, false);
              } else {
                break;
              }
            }
            frag_ptr[idx] = dst;
          } else {
            for (int i = 0; i < AccessSize; i++) {
              frag_ptr[idx][i] = static_cast<Element>(0);
            }
          }
          ++address_iterator_;
        }
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) { load_with_byte_offset(frag, 0); }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    store_with_byte_offset(frag, pointer_offset * sizeof_bits<Element>::value / 8);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, LongIndex byte_offset) {
    address_iterator_.set_iteration_index(0);
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {

          int idx = v + kAccessesPerVector * (c + s * ThreadMap::Iterations::kContiguous);

          char *byte_ptr = reinterpret_cast<char *>(address_iterator_.get()) + byte_offset;
          AccessType *access_ptr = reinterpret_cast<AccessType *>(byte_ptr);

          if (address_iterator_.valid()) {
            *access_ptr = frag_ptr[idx];
          }
          ++address_iterator_;
        }
      }
    }
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) { store_with_byte_offset(frag, 0); }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PositionPredicatedTileIterator for column-major data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int AccessSize
>
class PositionPredicatedTileIterator<Shape_, Element_, layout::ColumnMajor, AdvanceRank, ThreadMap_, AccessSize> {
public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1,
    "Specialization for pitch-linear iterator may along advance along the "
    "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PositionPredicatedTileIterator<
    layout::PitchLinearShape<Shape::kRow, Shape::kColumn>,
    Element,
    layout::PitchLinear,
    (kAdvanceRank == 0 ? 0 : 1),
    ThreadMap,
    AccessSize
  >;

  using AccessType = typename UnderlyingIterator::AccessType;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
  private:

    friend PositionPredicatedTileIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

  public:

    CUTLASS_HOST_DEVICE
    Params() { }

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout): params_(layout::PitchLinear(layout.stride(0))) {

    }
  };


private:

  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

public:

  /// Constructs a TileIterator from its precomputed state, threadblock offset, and thread ID
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator(
    Params const &params,                         ///< Precomputed parameters object
    Pointer pointer,                              ///< Pointer to start of tensor
    TensorCoord extent,                           ///< Extent of tensor
    int thread_id,                                ///< ID of each participating thread
    TensorCoord const &threadblock_offset         ///< Initial offset of threadblock
  ):
    iterator_(
      params.params_,
      pointer,
      layout::PitchLinearCoord(extent.row(), extent.column()),
      thread_id,
      layout::PitchLinearCoord(threadblock_offset.row(), threadblock_offset.column())
    ) { }

  /// Construct a PositionPredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator(
    Params const &params,                         ///< Precomputed parameters object
    Pointer pointer,                              ///< Pointer to start of tensor
    TensorCoord extent,                           ///< Extent of tensor
    int thread_id                                 ///< ID of each participating thread
  ): PositionPredicatedTileIterator(params, pointer, extent, thread_id, make_Coord(0, 0)) { }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator operator++(int) {
    PositionPredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask() {
    iterator_.clear_mask();
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) {
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
    iterator_.get_mask(mask);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, LongIndex byte_offset) {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, LongIndex byte_offset) {
    iterator_.store_with_byte_offset(frag, byte_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PositionPredicatedTileIterator for row-major data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  int AccessSize
>
class PositionPredicatedTileIterator<Shape_, Element_, layout::RowMajor, AdvanceRank, ThreadMap_, AccessSize> {
public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1,
    "Specialization for pitch-linear iterator may along advance along the "
    "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PositionPredicatedTileIterator<
    layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,
    Element,
    layout::PitchLinear,
    (kAdvanceRank == 0 ? 1 : 0),
    ThreadMap,
    AccessSize
  >;

  using AccessType = typename UnderlyingIterator::AccessType;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
  private:

    friend PositionPredicatedTileIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

  public:

    CUTLASS_HOST_DEVICE
    Params() { }

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout,
    int window_size,                              ///< window size to be sampled
    int channels = 1): params_(layout::PitchLinear(layout.stride(0)), window_size, channels) {

    };
  };


private:

  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

public:

  /// Constructs a TileIterator from its precomputed state, threadblock offset, and thread ID
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator(
    Params const &params,                         ///< Precomputed parameters object
    Pointer pointer,                              ///< Pointer to start of tensor
    TensorCoord extent,                           ///< Extent of tensor
    int thread_id,                                ///< ID of each participating thread
    TensorCoord const &threadblock_offset         ///< Initial offset of threadblock
  ):
    iterator_(
      params.params_,
      pointer,
      layout::PitchLinearCoord(extent.column(), extent.row()),
      thread_id,
      layout::PitchLinearCoord(threadblock_offset.column(), threadblock_offset.row())
    ) { }

  /// Construct a PositionPredicatedTileIterator with zero threadblock offset
  // CUTLASS_HOST_DEVICE
  // PositionPredicatedTileIterator(
  //   Params const &params,                         ///< Precomputed parameters object
  //   Pointer pointer,                              ///< Pointer to start of tensor
  //   TensorCoord extent,                           ///< Extent of tensor
  //   int thread_id                                 ///< ID of each participating thread
  // ): PositionPredicatedTileIterator(params, pointer, extent, thread_id, make_Coord(0, 0)) { }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator operator++(int) {
    PositionPredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask() {
    iterator_.clear_mask();
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) {
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
    iterator_.get_mask(mask);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, LongIndex byte_offset) {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, LongIndex byte_offset) {
    iterator_.store_with_byte_offset(frag, byte_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PositionPredicatedTileIterator for interleaved data.  It is mapped
/// to the congruous layout.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///

template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int AccessSize, int InterleavedK>
class PositionPredicatedTileIterator<Shape_, Element_,
                             layout::ColumnMajorInterleaved<InterleavedK>,
                             AdvanceRank, ThreadMap_, AccessSize> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  static int const kInterleavedK = InterleavedK;
  using Layout = layout::ColumnMajorInterleaved<kInterleavedK>;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PositionPredicatedTileIterator<
      layout::PitchLinearShape<Shape::kRow * kInterleavedK,
                               Shape::kColumn / kInterleavedK>,
      Element, layout::PitchLinear, (kAdvanceRank == 0 ? 0 : 1), ThreadMap, AccessSize>;


  using AccessType = typename UnderlyingIterator::AccessType;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                               ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PositionPredicatedTileIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:
    CUTLASS_HOST_DEVICE
    Params() {}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout)
        : params_(layout::PitchLinear(layout.stride(0))) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : iterator_(params.params_, pointer,
                  layout::PitchLinearCoord(extent.row() * kInterleavedK,
                                           extent.column() / kInterleavedK),
                  thread_id,
                  layout::PitchLinearCoord(
                      threadblock_offset.row() * kInterleavedK,
                      threadblock_offset.column() / kInterleavedK)) {}

  /// Construct a PositionPredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator(
      Params const &params,  ///< Precomputed parameters object
      Pointer pointer,       ///< Pointer to start of tensor
      TensorCoord extent,    ///< Extent of tensor
      int thread_id          ///< ID of each participating thread
      )
      : PositionPredicatedTileIterator(params, pointer, extent, thread_id,
                               make_Coord(0, 0)) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator operator++(int) {
    PositionPredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask() { iterator_.clear_mask(); }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() { iterator_.enable_mask(); }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { iterator_.set_mask(mask); }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { iterator_.get_mask(mask); }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) { load_with_pointer_offset(frag, 0); }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) { store_with_pointer_offset(frag, 0); }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PositionPredicatedTileIterator for interleaved-32 data.  It is
/// mapped to the congruous layout.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int AccessSize, int InterleavedK>
class PositionPredicatedTileIterator<Shape_, Element_,
                             layout::RowMajorInterleaved<InterleavedK>,
                             AdvanceRank, ThreadMap_, AccessSize> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  static int const kInterleavedK = InterleavedK;
  using Layout = layout::RowMajorInterleaved<kInterleavedK>;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PositionPredicatedTileIterator<
      layout::PitchLinearShape<Shape::kColumn * kInterleavedK,
                               Shape::kRow / kInterleavedK>,
      Element, layout::PitchLinear, (kAdvanceRank == 0 ? 1 : 0), ThreadMap, AccessSize>;


  using AccessType = typename UnderlyingIterator::AccessType;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                               ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend PositionPredicatedTileIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:
    CUTLASS_HOST_DEVICE
    Params() {}

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout)
        : params_(layout::PitchLinear(layout.stride(0))) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset)
      : iterator_(params.params_, pointer,
                  layout::PitchLinearCoord(extent.column() * kInterleavedK,
                                           extent.row() / kInterleavedK),
                  thread_id,
                  layout::PitchLinearCoord(
                      threadblock_offset.column() * kInterleavedK,
                      threadblock_offset.row() / kInterleavedK)) {}

  /// Construct a PositionPredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator(
      Params const &params,  ///< Precomputed parameters object
      Pointer pointer,       ///< Pointer to start of tensor
      TensorCoord extent,    ///< Extent of tensor
      int thread_id          ///< ID of each participating thread
      )
      : PositionPredicatedTileIterator(params, pointer, extent, thread_id,
                               make_Coord(0, 0)) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  PositionPredicatedTileIterator operator++(int) {
    PositionPredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask() { iterator_.clear_mask(); }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() { iterator_.enable_mask(); }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { iterator_.set_mask(mask); }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { iterator_.get_mask(mask); }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) { load_with_pointer_offset(frag, 0); }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) { store_with_pointer_offset(frag, 0); }
};

// ////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace transform
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
