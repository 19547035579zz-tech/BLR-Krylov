#pragma once
#include "blr_krylov_types.h"
#include <cublas_v2.h>

cudaError_t iter_compute_residual(
    cublasHandle_t cublas_h,
    const BLRMatrix& blr_A,
    const std::vector<float*>& d_U_decomp,
    const std::vector<float*>& d_Vt_decomp,
    const float* d_x,
    const float* d_b,
    float& res
);

void iter_adjust_r_threshold(
    float curr_res,
    const IterParams& params,
    int& curr_r
);

cudaError_t iter_main_loop(
    cublasHandle_t cublas_h,
    cusparseHandle_t cusparse_h,
    const BLRMatrix& blr_A,
    const std::vector<float*>& d_U_decomp,
    const std::vector<float*>& d_Vt_decomp,
    const float* d_V0,
    const float* d_b,
    float* d_x,
    const IterParams& params,
    size_t k
);

IterParams iter_init_params(
    float res_threshold = 1e-6,
    int r_init = 32,
    int r_max = 64,
    int r_min = 16,
    int max_iter = 100,
    float zfp_eps = 1e-6
);
