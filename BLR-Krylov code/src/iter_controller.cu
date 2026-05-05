#include "iter_controller.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cassert>

static cudaError_t compute_Ax(
    const BLRMatrix& blr_A,
    const std::vector<float*>& d_U_decomp,
    const std::vector<float*>& d_Vt_decomp,
    const float* d_x,
    size_t M,
    float* d_Ax
) {
    BasisVectors temp_basis;
    temp_basis.rows = M;
    temp_basis.cols = 1;
    temp_basis.curr_valid_cnt = 1;
    cudaMalloc(&temp_basis.d_V, M * 1 * sizeof(float));
    cudaMalloc(&temp_basis.d_valid, 1 * sizeof(bool));
    cudaMemcpy(temp_basis.d_V, d_x, M * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemset(temp_basis.d_valid, true, sizeof(bool));

    CUDA_CHECK(fused_spgemm_execute(
        blr_A, d_U_decomp, d_Vt_decomp, temp_basis, d_Ax
    ));

    ca_krylov_free_basis(temp_basis);
    return cudaSuccess;
}

cudaError_t iter_compute_residual(
    cublasHandle_t cublas_h,
    const BLRMatrix& blr_A,
    const std::vector<float*>& d_U_decomp,
    const std::vector<float*>& d_Vt_decomp,
    const float* d_x,
    const float* d_b,
    float& res
) {
    const size_t M = blr_A.mat_rows;
    float alpha = -1.0f, beta = 1.0f;

    float* d_Ax;
    cudaMalloc(&d_Ax, M * sizeof(float));
    CUDA_CHECK(compute_Ax(blr_A, d_U_decomp, d_Vt_decomp, d_x, M, d_Ax));

    cublasSaxpy(cublas_h, static_cast<int>(M), &alpha, d_b, 1, d_Ax, 1);

    cublasSnrm2(cublas_h, static_cast<int>(M), d_Ax, 1, &res);

    cudaFree(d_Ax);
    return cudaSuccess;
}

void iter_adjust_r_threshold(
    float curr_res,
    const IterParams& params,
    int& curr_r
) {
    if (curr_res > params.res_threshold) {
        curr_r = std::min(curr_r + 4, params.r_max);
    } else if (curr_res < params.res_threshold / 10.0f) {
        curr_r = std::max(curr_r - 4, params.r_min);
    }
}

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
) {
    const size_t M = blr_A.mat_rows;
    float curr_res = 0.0f;
    int curr_r = params.r_init;
    BasisVectors prev_basis, curr_basis;
    bool converged = false;

    cudaMemset(d_x, 0, M * sizeof(float));

    CUDA_CHECK(ca_krylov_compute_basis(
        blr_A, d_U_decomp, d_Vt_decomp, d_V0, k, nullptr, curr_basis
    ));

    for (int iter = 0; iter < params.max_iter; iter++) {
        float* d_C;
        cudaMalloc(&d_C, M * k * sizeof(float));
        CUDA_CHECK(fused_spgemm_execute(
            blr_A, d_U_decomp, d_Vt_decomp, curr_basis, d_C
        ));

        float alpha = 1.0f;
        cublasSaxpy(cublas_h, static_cast<int>(M), &alpha, &d_C[0], k, d_x, 1);

        CUDA_CHECK(iter_compute_residual(
            cublas_h, blr_A, d_U_decomp, d_Vt_decomp, d_x, d_b, curr_res
        ));

        printf("[Iter %d] Residual = %.2e, r_thresh = %d, Valid Basis = %d/%zu\n",
               iter, curr_res, curr_r, curr_basis.curr_valid_cnt, k);

        if (curr_res <= params.res_threshold) {
            converged = true;
            cudaFree(d_C);
            break;
        }

        iter_adjust_r_threshold(curr_res, params, curr_r);

        ca_krylov_free_basis(prev_basis);
        CUDA_CHECK(ca_krylov_reuse_basis(curr_basis, curr_res, params.res_threshold, prev_basis));

        ca_krylov_free_basis(curr_basis);
        CUDA_CHECK(ca_krylov_compute_basis(
            blr_A, d_U_decomp, d_Vt_decomp, d_V0, k, &prev_basis, curr_basis
        ));

        cudaFree(d_C);
    }

    ca_krylov_free_basis(prev_basis);
    ca_krylov_free_basis(curr_basis);

    if (!converged) {
        printf("Warning: Iteration did not converge (max iter reached), final residual = %.2e\n", curr_res);
        return cudaErrorTimeout;
    }
    return cudaSuccess;
}

IterParams iter_init_params(
    float res_threshold,
    int r_init,
    int r_max,
    int r_min,
    int max_iter,
    float zfp_eps
) {
    IterParams params;
    params.res_threshold = res_threshold;
    params.r_init = r_init;
    params.r_max = r_max;
    params.r_min = r_min;
    params.max_iter = max_iter;
    params.zfp_eps = zfp_eps;
    return params;
}
