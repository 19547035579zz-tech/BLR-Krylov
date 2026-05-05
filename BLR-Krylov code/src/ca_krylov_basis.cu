#include "ca_krylov_basis.h"
#include <cuda_runtime.h>
#include <cassert>

__global__ void matrix_power_kernel(
    const BLRMatrix blr_A,
    const float** d_U_decomp,
    const float** d_Vt_decomp,
    const float* d_V0,
    const float* d_prev_V,
    const bool* d_prev_valid,
    size_t k,
    float* d_V,
    bool* d_valid
) {
    const size_t block_idx = blockIdx.x;
    if (block_idx >= blr_A.total_blocks) return;

    const int p = blr_A.block_p[block_idx];
    const int q = blr_A.block_q[block_idx];
    const int B = blr_A.super_block_size;
    const int row_start = p * B;
    const int col_start = q * B;
    const size_t M = blr_A.mat_rows;

    __shared__ float sm_V[64][4];
    __shared__ bool sm_valid[4];

    const int tid = threadIdx.x;
    if (tid < B) {
        const int global_col = col_start + tid;
        if (global_col < static_cast<int>(M)) {
            sm_V[tid][0] = d_V0[global_col];
            for (size_t v = 1; v < k; v++) {
                if (d_prev_valid[v]) {
                    sm_V[tid][v] = d_prev_V[global_col * k + v];
                    sm_valid[v] = true;
                } else {
                    sm_V[tid][v] = 0.0f;
                    sm_valid[v] = false;
                }
            }
        }
    }
    __syncthreads();

    for (size_t v = 0; v < k - 1; v++) {
        if (!sm_valid[v] && v > 0) continue;

        float* V_curr = &sm_V[0][v];
        float* V_next = &sm_V[0][v + 1];

        if (blr_A.block_type[block_idx] == 1) {
            const float* d_U = d_U_decomp[block_idx];
            const float* d_Vt = d_Vt_decomp[block_idx];
            const int r = blr_A.r_list[block_idx];

            __shared__ float sm_VtV[32];
            if (tid < r) {
                float sum = 0.0f;
                for (int y = 0; y < B; y++) {
                    sum += d_Vt[tid * B + y] * V_curr[y];
                }
                sm_VtV[tid] = sum;
            }
            __syncthreads();

            if (tid < B) {
                float sum = 0.0f;
                for (int rt = 0; rt < r; rt++) {
                    sum += d_U[tid * r + rt] * sm_VtV[rt];
                }
                V_next[tid] = sum;
                sm_valid[v + 1] = true;
            }
        } else if (blr_A.block_type[block_idx] == 2) {
            const float* d_A_morton = blr_A.dense_morton[block_idx];
            if (tid < B) {
                float sum = 0.0f;
                for (int y = 0; y < B; y++) {
                    sum += d_A_morton[tid * B + y] * V_curr[y];
                }
                V_next[tid] = sum;
                sm_valid[v + 1] = true;
            }
        } else {
            if (tid < B) {
                V_next[tid] = 0.0f;
                sm_valid[v + 1] = false;
            }
        }
        __syncthreads();
    }

    if (tid < B) {
        const int global_row = row_start + tid;
        if (global_row < static_cast<int>(M)) {
            for (size_t v = 0; v < k; v++) {
                d_V[global_row * k + v] = sm_V[tid][v];
            }
        }
    }

    if (tid == 0) {
        for (size_t v = 0; v < k; v++) {
            d_valid[v] = sm_valid[v];
        }
    }
}

cudaError_t ca_krylov_compute_basis(
    const BLRMatrix& blr_A,
    const std::vector<float*>& d_U_decomp,
    const std::vector<float*>& d_Vt_decomp,
    const float* d_V0,
    size_t k,
    const BasisVectors* prev_basis,
    BasisVectors& curr_basis
) {
    const size_t M = blr_A.mat_rows;
    curr_basis.rows = M;
    curr_basis.cols = k;

    cudaMalloc(&curr_basis.d_V, M * k * sizeof(float));
    if (!curr_basis.d_V) return cudaErrorMemoryAllocation;

    CUDA_CHECK(utils::valid_map_init(k, curr_basis.d_valid, true));

    const float* d_prev_V = prev_basis ? prev_basis->d_V : nullptr;
    const bool* d_prev_valid = prev_basis ? prev_basis->d_valid : nullptr;

    float** d_U_decomp_dev;
    float** d_Vt_decomp_dev;
    cudaMalloc(&d_U_decomp_dev, d_U_decomp.size() * sizeof(float*));
    cudaMalloc(&d_Vt_decomp_dev, d_Vt_decomp.size() * sizeof(float*));
    cudaMemcpy(d_U_decomp_dev, d_U_decomp.data(), d_U_decomp.size() * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vt_decomp_dev, d_Vt_decomp.data(), d_Vt_decomp.size() * sizeof(float*), cudaMemcpyHostToDevice);

    dim3 grid(blr_A.total_blocks);
    dim3 block(blr_A.super_block_size);
    matrix_power_kernel<<<grid, block>>>(
        blr_A, d_U_decomp_dev, d_Vt_decomp_dev, d_V0,
        d_prev_V, d_prev_valid, k, curr_basis.d_V, curr_basis.d_valid
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(utils::valid_map_query_count(curr_basis.d_valid, k, curr_basis.curr_valid_cnt));

    cudaFree(d_U_decomp_dev);
    cudaFree(d_Vt_decomp_dev);

    return cudaSuccess;
}

cudaError_t ca_krylov_reuse_basis(
    const BasisVectors& prev_basis,
    float curr_res,
    float res_threshold,
    BasisVectors& curr_basis
) {
    if (prev_basis.cols == 0 || !prev_basis.d_V || !prev_basis.d_valid) {
        return cudaErrorInvalidDevicePointer;
    }

    curr_basis.rows = prev_basis.rows;
    curr_basis.cols = prev_basis.cols;

    cudaMalloc(&curr_basis.d_V, prev_basis.rows * prev_basis.cols * sizeof(float));
    cudaMalloc(&curr_basis.d_valid, prev_basis.cols * sizeof(bool));
    if (!curr_basis.d_V || !curr_basis.d_valid) return cudaErrorMemoryAllocation;

    cudaMemcpy(curr_basis.d_V, prev_basis.d_V, 
               prev_basis.rows * prev_basis.cols * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(curr_basis.d_valid, prev_basis.d_valid, 
               prev_basis.cols * sizeof(bool), cudaMemcpyDeviceToDevice);

    std::vector<int> invalid_indices;
    if (curr_res > res_threshold) {
        for (size_t v = 0; v < prev_basis.cols; v++) {
            bool valid;
            cudaMemcpy(&valid, &prev_basis.d_valid[v], sizeof(bool), cudaMemcpyDeviceToHost);
            if (valid) {
                invalid_indices.push_back(static_cast<int>(v));
                break;
            }
        }
        if (!invalid_indices.empty()) {
            CUDA_CHECK(utils::valid_map_batch_update(
                curr_basis.d_valid, invalid_indices, invalid_indices.size()
            ));
        }
    }

    CUDA_CHECK(utils::valid_map_query_count(curr_basis.d_valid, curr_basis.cols, curr_basis.curr_valid_cnt));

    return cudaSuccess;
}

void ca_krylov_free_basis(BasisVectors& basis) {
    basis.free();
    basis.rows = 0;
    basis.cols = 0;
    basis.curr_valid_cnt = 0;
}
