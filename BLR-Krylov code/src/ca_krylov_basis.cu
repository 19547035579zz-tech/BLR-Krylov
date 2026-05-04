#include "ca_krylov_basis.h"
#include <cuda_runtime.h>
#include <cassert>

// GPU端矩阵幂核（论文4.4节Algorithm 2）
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

    // 共享内存：缓存基向量（V0~Vk-1）
    __shared__ float sm_V[64][4];  // 64×4（B=64，k=4）
    __shared__ bool sm_valid[4];   // 基向量有效性标记

    // 步骤1：批量加载基向量（V0 + 复用的有效向量）
    const int tid = threadIdx.x;
    if (tid < B) {
        const int global_col = col_start + tid;
        if (global_col < static_cast<int>(M)) {
            // 加载V0
            sm_V[tid][0] = d_V0[global_col];
            // 加载上一轮有效向量
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

    // 步骤2：迭代计算V1~Vk-1
    for (size_t v = 0; v < k - 1; v++) {
        if (!sm_valid[v] && v > 0) continue;  // 跳过无效向量

        float* V_curr = &sm_V[0][v];
        float* V_next = &sm_V[0][v + 1];

        if (blr_A.block_type[block_idx] == 1) {
            // 低秩块：U × (V^T × V_curr)
            const float* d_U = d_U_decomp[block_idx];
            const float* d_Vt = d_Vt_decomp[block_idx];
            const int r = blr_A.r_list[block_idx];

            // 计算V^T × V_curr（r×1）
            __shared__ float sm_VtV[32];  // r≤32
            if (tid < r) {
                float sum = 0.0f;
                for (int y = 0; y < B; y++) {
                    sum += d_Vt[tid * B + y] * V_curr[y];
                }
                sm_VtV[tid] = sum;
            }
            __syncthreads();

            // 计算U × (V^T × V_curr)（B×1）
            if (tid < B) {
                float sum = 0.0f;
                for (int rt = 0; rt < r; rt++) {
                    sum += d_U[tid * r + rt] * sm_VtV[rt];
                }
                V_next[tid] = sum;
                sm_valid[v + 1] = true;  // 标记为有效
            }
        } else if (blr_A.block_type[block_idx] == 2) {
            // 稠密块：张量核计算（简化版FP32，实际需调用mma.sync）
            const float* d_A_morton = blr_A.dense_morton[block_idx];
            if (tid < B) {
                float sum = 0.0f;
                for (int y = 0; y < B; y++) {
                    // 简化：Morton地址映射（实际需按编码查找）
                    sum += d_A_morton[tid * B + y] * V_curr[y];
                }
                V_next[tid] = sum;
                sm_valid[v + 1] = true;
            }
        } else {
            // 零块：输出0
            if (tid < B) {
                V_next[tid] = 0.0f;
                sm_valid[v + 1] = false;
            }
        }
        __syncthreads();
    }

    // 步骤3：批量存储基向量至全局内存
    if (tid < B) {
        const int global_row = row_start + tid;
        if (global_row < static_cast<int>(M)) {
            for (size_t v = 0; v < k; v++) {
                d_V[global_row * k + v] = sm_V[tid][v];
            }
        }
    }

    // 步骤4：更新有效性标记（仅线程0）
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

    // 步骤1：分配当前基向量内存
    cudaMalloc(&curr_basis.d_V, M    * k * sizeof(float));
    if (!curr_basis.d_V) return cudaErrorMemoryAllocation;

    // 步骤2：初始化有效性标记
    CUDA_CHECK(utils::valid_map_init(k, curr_basis.d_valid, true));

    // 步骤3：准备上一轮基向量（若有）
    const float* d_prev_V = prev_basis ? prev_basis->d_V : nullptr;
    const bool* d_prev_valid = prev_basis ? prev_basis->d_valid : nullptr;

    // 步骤4：转换U/V^T列表为设备端指针数组（供核函数访问）
    float** d_U_decomp_dev;
    float** d_Vt_decomp_dev;
    cudaMalloc(&d_U_decomp_dev, d_U_decomp.size() * sizeof(float*));
    cudaMalloc(&d_Vt_decomp_dev, d_Vt_decomp.size() * sizeof(float*));
    cudaMemcpy(d_U_decomp_dev, d_U_decomp.data(), d_U_decomp.size() * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vt_decomp_dev, d_Vt_decomp.data(), d_Vt_decomp.size() * sizeof(float*), cudaMemcpyHostToDevice);

    // 步骤5：调用矩阵幂核批量计算基向量
    dim3 grid(blr_A.total_blocks);                  // 每个超级块1个线程块
    dim3 block(blr_A.super_block_size);              // 每个线程处理超级块1行
    matrix_power_kernel<<<grid, block>>>(
        blr_A, d_U_decomp_dev, d_Vt_decomp_dev, d_V0,
        d_prev_V, d_prev_valid, k, curr_basis.d_V, curr_basis.d_valid
    );
    CUDA_CHECK(cudaGetLastError());

    // 步骤6：查询有效基向量数量
    CUDA_CHECK(utils::valid_map_query_count(curr_basis.d_valid, k, curr_basis.curr_valid_cnt));

    // 步骤7：释放临时设备指针数组
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

    // 步骤1：分配当前基向量内存
    cudaMalloc(&curr_basis.d_V, prev_basis.rows * prev_basis.cols * sizeof(float));
    cudaMalloc(&curr_basis.d_valid, prev_basis.cols * sizeof(bool));
    if (!curr_basis.d_V || !curr_basis.d_valid) return cudaErrorMemoryAllocation;

    // 步骤2：拷贝上一轮基向量数据
    cudaMemcpy(curr_basis.d_V, prev_basis.d_V, 
               prev_basis.rows * prev_basis.cols * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(curr_basis.d_valid, prev_basis.d_valid, 
               prev_basis.cols * sizeof(bool), cudaMemcpyDeviceToDevice);

    // 步骤3：若残差超标，标记最旧的基向量为无效（论文4.5节复用逻辑）
    std::vector<int> invalid_indices;
    if (curr_res > res_threshold) {
        // 查找第一个有效基向量并标记无效
        for (size_t v = 0; v < prev_basis.cols; v++) {
            bool valid;
            cudaMemcpy(&valid, &prev_basis.d_valid[v], sizeof(bool), cudaMemcpyDeviceToHost);
            if (valid) {
                invalid_indices.push_back(static_cast<int>(v));
                break;
            }
        }
        // 批量更新ValidMap
        if (!invalid_indices.empty()) {
            CUDA_CHECK(utils::valid_map_batch_update(
                curr_basis.d_valid, invalid_indices, invalid_indices.size()
            ));
        }
    }

    // 步骤4：更新有效基向量数量
    CUDA_CHECK(utils::valid_map_query_count(curr_basis.d_valid, curr_basis.cols, curr_basis.curr_valid_cnt));

    return cudaSuccess;
}

void ca_krylov_free_basis(BasisVectors& basis) {
    basis.free();
    basis.rows = 0;
    basis.cols = 0;
    basis.curr_valid_cnt = 0;
}