#pragma once
#include "blr_krylov_types.h"
#include "utils/valid_map.cuh"

cudaError_t ca_krylov_compute_basis(
    const BLRMatrix& blr_A,
    const std::vector<float*>& d_U_decomp,
    const std::vector<float*>& d_Vt_decomp,
    const float* d_V0,
    size_t k,
    const BasisVectors* prev_basis,
    BasisVectors& curr_basis
);

cudaError_t ca_krylov_reuse_basis(
    const BasisVectors& prev_basis,
    float curr_res,
    float res_threshold,
    BasisVectors& curr_basis
);

void ca_krylov_free_basis(BasisVectors& basis);
