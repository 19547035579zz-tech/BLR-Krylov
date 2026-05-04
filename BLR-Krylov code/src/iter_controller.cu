#include "iter_controller.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cassert>

// 计算Ax = A × x（调用fused_spgemm）
static cudaError_t compute_Ax(
    const BLRMatrix& blr_A,
    const std::vector<float*>& d_U_decomp,
    const std::vector<float*>& d_Vt_decomp,
    const float* d_x,
    size_t M,
    float* d_Ax
) {
    // 构造临时基向量（x作为单个基向量，k=1）
    BasisVectors temp_basis;
    temp_basis.rows = M;
    temp_basis.cols = 1;
    temp_basis.curr_valid_cnt = 1;
    cudaMalloc(&temp_basis.d_V, M * 1 * sizeof(float));
    cudaMalloc(&temp_basis.d_valid, 1 * sizeof(bool));
    cudaMemcpy(temp_basis.d_V, d_x, M * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemset(temp_basis.d_valid, true, sizeof(bool));

    // 调用fused_spgemm计算Ax = A × x
    CUDA_CHECK(fused_spgemm_execute(
        blr_A, d_U_decomp, d_Vt_decomp, temp_basis, d_Ax
    ));

    // 释放临时基向量
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

    // 步骤1：计算Ax = A × x
    float* d_Ax;
    cudaMalloc(&d_Ax, M * sizeof(float));
    CUDA_CHECK(compute_Ax(blr_A, d_U_decomp, d_Vt_decomp, d_x, M, d_Ax));

    // 步骤2：计算Ax - b（d_Ax = d_Ax - d_b）
    cublasSaxpy(cublas_h, static_cast<int>(M), &alpha, d_b, 1, d_Ax, 1);

    // 步骤3：计算L2残差（res = ||Ax - b||₂）
    cublasSnrm2(cublas_h, static_cast<int>(M), d_Ax, 1, &res);

    // 步骤4：释放临时内存
    cudaFree(d_Ax);
    return cudaSuccess;
}

void iter_adjust_r_threshold(
    float curr_res,
    const IterParams& params,
    int& curr_r
) {
    if (curr_res > params.res_threshold) {
        // 残差超标：提升秩阈值（最多至r_max）
        curr_r = std::min(curr_r + 4, params.r_max);
    } else if (curr_res < params.res_threshold / 10.0f) {
        // 残差过小：降低秩阈值（最少至r_min）
        curr_r = std::max(curr_r - 4, params.r_min);
    }
    // 残差在合理范围：保持当前秩阈值
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

    // 步骤1：初始化解向量x为0
    cudaMemset(d_x, 0, M * sizeof(float));

    // 步骤2：第一轮基向量计算（无复用）
    CUDA_CHECK(ca_krylov_compute_basis(
        blr_A, d_U_decomp, d_Vt_decomp, d_V0, k, nullptr, curr_basis
    ));

    // 步骤3：迭代主循环
    for (int iter = 0; iter < params.max_iter; iter++) {
        // 3.1 计算SpGEMM结果C = A × V（基向量矩阵）
        float* d_C;
        cudaMalloc(&d_C, M * k * sizeof(float));
        CUDA_CHECK(fused_spgemm_execute(
            blr_A, d_U_decomp, d_Vt_decomp, curr_basis, d_C
        ));

        // 3.2 更新解向量x（简化：x = x + C[:, 0]，实际需Krylov子空间投影）
        float alpha = 1.0f;
        cublasSaxpy(cublas_h, static_cast<int>(M), &alpha, &d_C[0], k, d_x, 1);

        // 3.3 计算当前残差
        CUDA_CHECK(iter_compute_residual(
            cublas_h, blr_A, d_U_decomp, d_Vt_decomp, d_x, d_b, curr_res
        ));

        // 3.4 输出迭代日志
        printf("[Iter %d] Residual = %.2e, r_thresh = %d, Valid Basis = %d/%zu\n",
               iter, curr_res, curr_r, curr_basis.curr_valid_cnt, k);

        // 3.5 检查收敛
        if (curr_res <= params.res_threshold) {
            converged = true;
            cudaFree(d_C);
            break;
        }

        // 3.6 动态调整低秩阈值（下一轮预处理生效，此处简化）
        iter_adjust_r_threshold(curr_res, params, curr_r);

        // 3.7 基向量复用（准备下一轮）
        ca_krylov_free_basis(prev_basis);
        CUDA_CHECK(ca_krylov_reuse_basis(curr_basis, curr_res, params.res_threshold, prev_basis));

        // 3.8 下一轮基向量计算（复用prev_basis）
        ca_krylov_free_basis(curr_basis);
        CUDA_CHECK(ca_krylov_compute_basis(
            blr_A, d_U_decomp, d_Vt_decomp, d_V0, k, &prev_basis, curr_basis
        ));

        // 3.9 释放临时结果C
        cudaFree(d_C);
    }

    // 步骤4：释放基向量内存
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