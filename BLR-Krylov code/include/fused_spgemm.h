#pragma once
#include "blr_krylov_types.h"

cudaError_t fused_spgemm_lowrank_block(
    const float* d_U,
    const float* d_Vt,
    const float* d_V,
    int B,
    int r,
    size_t k,
    float* d_C_block
);

cudaError_t fused_spgemm_dense_block_tensor_core(
    const float* d_A_morton,
    const float* d_V,
    int B,
    size_t k,
    float* d_C_block
);

cudaError_t fused_spgemm_execute(
    const BLRMatrix& blr_A,
    const std::vector<float*>& d_U_decomp,
    const std::vector<float*>& d_Vt_decomp,
    const BasisVectors& basis,
    float* d_C
);

cudaError_t fused_spgemm_aggregate(
    const std::vector<float*>& d_C_blocks,
    const BLRMatrix& blr_A,
    size_t k,
    float* d_C
);
