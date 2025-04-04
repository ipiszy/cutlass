/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cute/tensor.hpp"
#include "cute/layout.hpp"

#include "collective/fmha_common.hpp"
#include "collective/fmha_fusion.hpp"

namespace cutlass::fmha::collective {

static constexpr int kSFBlockSize = 32;
using namespace cute;

template<
  class Element,
  class StrideQ,
  class StrideK,
  class StrideV,
  class CollectiveMmaQK,
  class CollectiveMmaPV,
  class SmemLayoutQ,
  class SmemLayoutK,
  class SmemLayoutV,
  class TensorStorage,
  class PipelineQ,
  class PipelineKV,
  class Mask,
  class TileShape
#ifdef MXFP8
  ,
  class SFElement,
  class StrideSFQ,
  class StrideSFK,
  class StrideSFV,
  class SmemLayoutSFQ,
  class SmemLayoutSFK,
  class SmemLayoutSFV
#endif
>
struct Sm100FmhaLoadTmaWarpspecialized {

  using TileShapeQK = typename CollectiveMmaQK::TileShape;
  using TileShapePV = typename CollectiveMmaPV::TileShape;
  using TileShapeSFQK = decltype(make_shape(get<0>(TileShapeQK{}), get<1>(TileShapeQK{}), get<2>(TileShapeQK{}) / kSFBlockSize));
  using TileShapeSFPV = decltype(make_shape(get<0>(TileShapePV{}), get<1>(TileShapePV{}), get<2>(TileShapePV{}) / kSFBlockSize));

  struct Arguments {
    const Element* ptr_Q;
    StrideQ dQ;
    const Element* ptr_K;
    StrideK dK;
    const Element* ptr_V;
    StrideV dV;

#ifdef MXFP8
    const SFElement* ptr_SFQ;
    StrideSFQ dSFQ;
    const SFElement* ptr_SFK;
    StrideSFK dSFK;
    const SFElement* ptr_SFV;
    StrideSFV dSFV;
#endif
  };

  using TMA_Q = typename CollectiveMmaQK::Params::TMA_A;
  using TMA_K = typename CollectiveMmaQK::Params::TMA_B;
  using TMA_V = typename CollectiveMmaPV::Params::TMA_B;

#ifdef MXFP8
  using TMA_SFQ = typename CollectiveMmaQK::Params::TMA_SFA;
  using TMA_SFK = typename CollectiveMmaQK::Params::TMA_SFB;
  using TMA_SFV = typename CollectiveMmaPV::Params::TMA_SFB;
#endif

  struct Params {
    TMA_Q tma_load_q;
    TMA_K tma_load_k;
    TMA_V tma_load_v;

#ifdef MXFP8
    TMA_SFQ tma_load_sfq;
    TMA_SFK tma_load_sfk;
    TMA_SFV tma_load_sfv;
#endif
  };

  template<class ProblemShape>
  static Params to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      void* workspace) {

    auto ptr_Q = args.ptr_Q;
    auto ptr_K = args.ptr_K;
    auto ptr_V = args.ptr_V;
    auto dQ = args.dQ;
    auto dK = args.dK;
    auto dV = args.dV;
    auto problem_shape_qk = problem_shape;

#ifdef MXFP8
    auto ptr_SFQ = args.ptr_SFQ;
    auto ptr_SFK = args.ptr_SFK;
    auto ptr_SFV = args.ptr_SFV;
    auto dSFQ = args.dSFQ;
    auto dSFK = args.dSFK;
    auto dSFV = args.dSFV;
#endif

    if constexpr (is_variable_length_v<tuple_element_t<0, ProblemShape>>) {
      auto cumulative_length_q = get<0>(problem_shape).cumulative_length;
      if (cumulative_length_q != nullptr) {
          int max_length_q = get<0>(problem_shape).max_length;
          // for variable sequence lenght, the batch is in units of row_stride
          get<2,1>(dQ) = get<0>(dQ);
          get<3,1>(problem_shape_qk) = std::max(get<3,1>(problem_shape_qk), max_length_q * (1 + get<3,1>(problem_shape)));
          // offset ptr by the amount we add back in later
          ptr_Q -= max_length_q * get<0>(dQ);
      }
    }

    if constexpr (is_variable_length_v<tuple_element_t<1, ProblemShape>>) {
      auto cumulative_length_kv = get<1>(problem_shape).cumulative_length;
      if (cumulative_length_kv != nullptr) {
          int max_length_kv = get<1>(problem_shape).max_length;
          // for variable sequence lenght, the batch is in units of row_stride
          get<2,1>(dK) = get<0>(dK);
          get<2,1>(dV) = get<0>(dV);
          get<3,1>(problem_shape_qk) = std::max(get<3,1>(problem_shape_qk), max_length_kv * (1 + get<3,1>(problem_shape)));
          // offset ptr by the amount we add back in later
          ptr_K -= max_length_kv * get<0>(dK);
          ptr_V -= max_length_kv * get<0>(dV);
      }
    }

    auto params_qk = CollectiveMmaQK::to_underlying_arguments(
        problem_shape_qk,
        typename CollectiveMmaQK::Arguments {
            ptr_Q, dQ,
            ptr_K, dK
#ifdef MXFP8
            ,
            ptr_SFQ, make_layout(make_shape(get<0>(TileShapeSFQK{}), get<2>(TileShapeSFQK{})), dSFQ),
            ptr_SFK, make_layout(make_shape(get<1>(TileShapeSFQK{}), get<2>(TileShapeSFQK{})), dSFK)
#endif
        }, /*workspace=*/ nullptr);

    auto problem_shape_pv = select<0,2,1,3>(problem_shape_qk);
    auto params_pv = CollectiveMmaPV::to_underlying_arguments(
        problem_shape_pv,
        typename CollectiveMmaPV::Arguments {
            ptr_K, dK,  // never used, dummy
            ptr_V, select<1,0,2>(dV),
#ifdef MXFP8
            ,
            ptr_SFQ, make_layout(make_shape(get<0>(TileShapeSFQK{}), get<2>(TileShapeSFQK{})), dSFQ),  // dummy
            ptr_SFV, make_layout(make_shape(get<2>(TileShapeSFPV{}), get<1>(TileShapeSFPV{})), dSFV)
#endif
        }, /*workspace=*/ nullptr);

    return Params{
        params_qk.tma_load_a,
        params_qk.tma_load_b,
        params_pv.tma_load_b
#ifdef MXFP8
        ,
        params_qk.tma_load_sfa,
        params_qk.tma_load_sfb,
        params_pv.tma_load_sfb
#endif
    };
  }


  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_k.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_v.get_tma_descriptor());
#ifdef MXFP8
    cute::prefetch_tma_descriptor(params.tma_load_sfq.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_sfk.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_sfv.get_tma_descriptor());
#endif
  }

  template<class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE void
  load(
      BlkCoord const& blk_coord_in, ProblemShape const& problem_shape,
      Params const& params, ParamsProblemShape const& params_problem_shape,
      TensorStorage& storage,
      PipelineQ& pipeline_q, typename PipelineQ::PipelineState& pipeline_q_producer_state,
      PipelineKV& pipeline_kv, typename PipelineKV::PipelineState& pipeline_kv_producer_state) {

    BlkCoord blk_coord_q = blk_coord_in;
    BlkCoord blk_coord_kv = blk_coord_in;

    int mask_tile_count = Mask{}.get_trip_count(blk_coord_in, TileShape{}, problem_shape);

    using X = Underscore;

    // this one is only executed by one thread, no need to elect_one

    // Q1, K1, Q2, V1, K2, V2, K3, V3, ...
    // two pipes: Q and KV
    // from Memory (prod) to TensorCore (cons)

    // compute gQ, sQ
    // we load 2*get<0>(blk_coord), and 2*get<0>(blk_coord) + 1
    ThrMMA mma_qk = typename CollectiveMmaQK::TiledMma{}.get_slice(0);
    Tensor mQ_qdl_p = params.tma_load_q.get_tma_tensor(select<0,2,3>(problem_shape));

    int q_offs_0 = 0;
    int q_offs_2_1 = 0;

    if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>) {
      auto cumulative_length_q = get<0>(params_problem_shape).cumulative_length;
      if (cumulative_length_q != nullptr) {
        int max_length_q = get<0>(params_problem_shape).max_length;
        q_offs_0 = max_length_q - get<0>(problem_shape);
        q_offs_2_1 = cumulative_length_q[get<2,1>(blk_coord_q)] + get<0>(problem_shape);
        get<2,1>(blk_coord_q) = 0;
      }
    }

    Tensor mQ_qdl = domain_offset(make_coord(q_offs_0, _0{}, make_coord(_0{}, q_offs_2_1)), mQ_qdl_p);

    Tensor gQ_qdl = local_tile(mQ_qdl, TileShapeQK{}, make_coord(_, _, _), Step<_1, X, _1>{});
    Tensor tSgQ_qdl = mma_qk.partition_A(gQ_qdl);
    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    auto [tQgQ_qdl, tQsQ] = tma_partition(
      params.tma_load_q, _0{}, make_layout(_1{}),
      group_modes<0,3>(sQ), group_modes<0,3>(tSgQ_qdl)
    );
    Tensor tQgQ = tQgQ_qdl(_, _, _0{}, get<2>(blk_coord_q));

#ifdef MXFP8
    Tensor mSFQ_qdl_p = params.tma_load_sfq.get_tma_tensor(
        make_shape(get<0>(problem_size), get<2>(problem_size) / kSFBlockSize, get<3>(problem_size)));
    Tensor mSFQ_qdl = domain_offset(make_coord(q_offs_0, _0{}, make_coord(_0{}, q_offs_2_1)), mSFQ_qdl_p);
    Tensor gSFQ_qdl = local_tile(mSFQ_qdl, TileShapeSFQK{}, make_coord(_, _, _), Step<_1, X, _1>{});
    Tensor tSgSFQ_qdl = mma_qk.partition_A(gSFQ_qdl);
    Tensor sSFQ = make_tensor(make_smem_ptr(storage.smem_sfq.data()), SmemLayoutQ{});

    auto [tSFQgQ_qdl, tSFQsQ] = tma_partition(
      params.tma_load_sfq, _0{}, make_layout(_1{}),
      group_modes<0,3>(sSFQ), group_modes<0,3>(tSgSFQ_qdl)
    );
    Tensor tSFQgQ = tSFQgQ_qdl(_, _, _0{}, get<2>(blk_coord_q));
#endif

    // compute gK, sK
    Tensor mK_kdl_p = params.tma_load_k.get_tma_tensor(select<1,2,3>(problem_shape));

    int kv_offs_0 = 0;
    int kv_offs_2_1 = 0;

    if constexpr (is_variable_length_v<tuple_element_t<1, ParamsProblemShape>>) {
      auto cumulative_length = get<1>(params_problem_shape).cumulative_length;
      if (cumulative_length != nullptr) {
        int max_length = get<1>(params_problem_shape).max_length;
        kv_offs_0 = max_length - get<1>(problem_shape);
        kv_offs_2_1 = cumulative_length[get<2,1>(blk_coord_kv)] + get<1>(problem_shape);
        get<2,1>(blk_coord_kv) = 0;
      }
    }

    Tensor mK_kdl = domain_offset(make_coord(kv_offs_0, _0{}, make_coord(_0{}, kv_offs_2_1)), mK_kdl_p);

    Tensor gK_kdl = local_tile(mK_kdl, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
    Tensor tSgK_kdl = mma_qk.partition_B(gK_kdl);
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
    auto [tKgK_kdl, tKsK] = tma_partition(
      params.tma_load_k, _0{}, make_layout(_1{}),
      group_modes<0,3>(sK), group_modes<0,3>(tSgK_kdl)
    );
    Tensor tKgK = tKgK_kdl(_, _, _0{}, get<2>(blk_coord_kv));

#ifdef MXFP8
    Tensor mSFK_kdl_p = params.tma_load_sfk.get_tma_tensor(
        make_shape(get<1>(problem_size), get<2>(problem_size) / kSFBlockSize, get<3>(problem_size)));

    // if constexpr (is_variable_length_v<tuple_element_t<1, ParamsProblemShape>>) {
    //   auto cumulative_length = get<1>(params_problem_shape).cumulative_length;
    //   if (cumulative_length != nullptr) {
    //     int max_length = get<1>(params_problem_shape).max_length;
    //     kv_offs_0 = max_length - get<1>(problem_shape);
    //     kv_offs_2_1 = cumulative_length[get<2,1>(blk_coord_kv)] + get<1>(problem_shape);
    //     get<2,1>(blk_coord_kv) = 0;
    //   }
    // }

    Tensor mSFK_kdl = domain_offset(make_coord(kv_offs_0, _0{}, make_coord(_0{}, kv_offs_2_1)), mSFK_kdl_p);

    Tensor gSFK_kdl = local_tile(mSFK_kdl, TileShapeSFQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
    Tensor tSgSFK_kdl = mma_qk.partition_B(gSFK_kdl);
    Tensor sSFK = make_tensor(make_smem_ptr(storage.smem_sfk.data()), SmemLayoutSFK{});
    auto [tSFKgK_kdl, tSFKsK] = tma_partition(
      params.tma_load_sfk, _0{}, make_layout(_1{}),
      group_modes<0,3>(sSFK), group_modes<0,3>(tSgSFK_kdl)
    );
    Tensor tSFKgK = tSFKgK_kdl(_, _, _0{}, get<2>(blk_coord_kv));
#endif

    // compute gV, sV
    ThrMMA mma_pv = typename CollectiveMmaPV::TiledMma{}.get_slice(0);
    Tensor mV_dkl_p = params.tma_load_v.get_tma_tensor(select<2,1,3>(problem_shape));

    Tensor mV_dkl = domain_offset(make_coord(_0{}, kv_offs_0, make_coord(_0{}, kv_offs_2_1)), mV_dkl_p);

    Tensor gV_dkl = local_tile(mV_dkl, TileShapePV{}, make_coord(_, _, _), Step<X, _1, _1>{});
    Tensor tOgV_dkl = mma_pv.partition_B(gV_dkl);
    Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});
    auto [tVgV_dkl, tVsV] = tma_partition(
      params.tma_load_v, _0{}, make_layout(_1{}),
      group_modes<0,3>(sV), group_modes<0,3>(tOgV_dkl)
    );
    auto tVgV = tVgV_dkl(_, _0{}, _, get<2>(blk_coord_kv));

#ifdef MXFP8
    Tensor mSFV_dkl_p = params.tma_load_sfv.get_tma_tensor(select<2,1,3>(problem_shape));

    Tensor mSFV_dkl = domain_offset(make_coord(_0{}, kv_offs_0, make_coord(_0{}, kv_offs_2_1)), mSFV_dkl_p);

    Tensor gSFV_dkl = local_tile(mSFV_dkl, TileShapeSFPV{}, make_coord(_, _, _), Step<X, _1, _1>{});
    Tensor tOgSFV_dkl = mma_pv.partition_B(gSFV_dkl);
    Tensor sSFV = make_tensor(make_smem_ptr(storage.smem_sfv.data()), SmemLayoutSFV{});
    auto [tSFVgV_dkl, tSFVsV] = tma_partition(
      params.tma_load_sfv, _0{}, make_layout(_1{}),
      group_modes<0,3>(sSFV), group_modes<0,3>(tOgSFV_dkl)
    );
    auto tSFVgV = tSFVgV_dkl(_, _0{}, _, get<2>(blk_coord_kv));
#endif

    // blk_coord in decomposed in terms of TileShape, not TileShapeQK
    // As such, it needs to be transformed as
    // (a,b,c): a -> 2*a (Q0) 2*a+1 (Q1)
    //          b -> 2*a (Ki i even) 2*a+1 (Ki i odd)

    uint32_t lane_predicate = cute::elect_one_sync();

    // Q1
    int q0_index = 2 * get<0>(blk_coord_q);
    int q1_index = 2 * get<0>(blk_coord_q) + 1;
    pipeline_q.producer_acquire(pipeline_q_producer_state);
    if (lane_predicate) {
      auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
      copy(params.tma_load_q.with(*tma_barrier, 0), tQgQ(_, q0_index), tQsQ(_, pipeline_q_producer_state.index()));
#ifdef MXFP8
      copy(params.tma_load_sfq.with(*tma_barrier, 0), tSFQgQ(_, q0_index), tSFQsQ(_, pipeline_q_producer_state.index()));
#endif
    }
    ++pipeline_q_producer_state;

    // K1
    int k_index = 0;
    pipeline_kv.producer_acquire(pipeline_kv_producer_state);
    if (lane_predicate) {
      auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
      copy(params.tma_load_k.with(*tma_barrier, 0), tKgK(_, k_index), tKsK(_, pipeline_kv_producer_state.index()));
#ifdef MXFP8
      copy(params.tma_load_sfk.with(*tma_barrier, 0), tSFKgK(_, k_index), tSFKsK(_, pipeline_kv_producer_state.index()));
#endif
    }
    ++pipeline_kv_producer_state;

    // Q2
    pipeline_q.producer_acquire(pipeline_q_producer_state);
    if (lane_predicate) {
      auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
      copy(params.tma_load_q.with(*tma_barrier, 0), tQgQ(_, q1_index), tQsQ(_, pipeline_q_producer_state.index()));
#ifdef MXFP8
      copy(params.tma_load_sfq.with(*tma_barrier, 0), tSFQgQ(_, q1_index), tSFQsQ(_, pipeline_q_producer_state.index()));
#endif
    }
    ++pipeline_q_producer_state;

    // V1
    pipeline_kv.producer_acquire(pipeline_kv_producer_state);
    if (lane_predicate) {
      auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
      copy(params.tma_load_v.with(*tma_barrier, 0), tVgV(_, k_index), tVsV(_, pipeline_kv_producer_state.index()));
#ifdef MXFP8
      copy(params.tma_load_sfv.with(*tma_barrier, 0), tSFVgV(_, k_index), tSFVsV(_, pipeline_kv_producer_state.index()));
#endif
    }
    ++pipeline_kv_producer_state;
    k_index += 1;

    // loop:
    mask_tile_count -= 1;
    for (; mask_tile_count > 0; mask_tile_count -= 1) {

      // Ki
      pipeline_kv.producer_acquire(pipeline_kv_producer_state);
      if (lane_predicate) {
        auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
        copy(params.tma_load_k.with(*tma_barrier, 0), tKgK(_, k_index), tKsK(_, pipeline_kv_producer_state.index()));
#ifdef MXFP8
        copy(params.tma_load_sfk.with(*tma_barrier, 0), tSFKgK(_, k_index), tSFKsK(_, pipeline_kv_producer_state.index()));
#endif
      }
      ++pipeline_kv_producer_state;

      // Vi
      pipeline_kv.producer_acquire(pipeline_kv_producer_state);
      if (lane_predicate) {
        auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
        copy(params.tma_load_v.with(*tma_barrier, 0), tVgV(_, k_index), tVsV(_, pipeline_kv_producer_state.index()));
#ifdef MXFP8
        copy(params.tma_load_sfv.with(*tma_barrier, 0), tSFVgV(_, k_index), tSFVsV(_, pipeline_kv_producer_state.index()));
#endif
      }
      ++pipeline_kv_producer_state;
      k_index += 1;
    }
  }
};

}  // namespace cutlass::fmha::collective
