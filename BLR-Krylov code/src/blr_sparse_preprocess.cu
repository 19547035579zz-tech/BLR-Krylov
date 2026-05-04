#include "blr_sparse_preprocess.h"
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cassert>
#include <fstream>
#include <iostream>

// GPU端交叉近似核（计算低秩块基向量，论文4.2节）
__global__ void cross_approximation_kernel(
    const float* d_A_block,
    int B,
    float eps,
    int r_thresh,
    float* d_U,
    float* d_Vt,
    int* d_r
) {
    const int tid = threadIdx.x;
    __shared__ float shared_A[64][64];  // 64×64超级块缓存
    __shared__ int shared_pivot[2];     // 主行/主列索引
    __shared__ float shared_norm[64];   // 列范数

    // 步骤1：加载超级块至共享内存
    if (tid < B * B) {
        const int x = tid / B;
        const int y = tid % B;
        shared_A[x][y] = d_A_block[tid];
    }
    __syncthreads();

    // 步骤2：选择主列（范数最大的列）
    if (tid < B) {
        float norm = 0.0f;
        for (int x = 0; x < B; x++) {
            norm += shared_A[x][tid] * shared_A[x][tid];
        }
        shared_norm[tid] = norm;
    }
    __syncthreads();

    // 步骤3：归约找最大范数列
    if (tid == 0) {
        float max_norm = 0.0f;
        int pivot_col = 0;
        for (int y = 0; y < B; y++) {
            if (shared_norm[y] > max_norm) {
                max_norm = shared_norm[y];
                pivot_col = y;
            }
        }
        shared_pivot[0] = pivot_col;  // 主列
        shared_pivot[1] = 0;          // 初始主行
    }
    __syncthreads();

    // 步骤4：构造U和V^T（简化版，实际需迭代优化）
    const int pivot_col = shared_pivot[0];
    const int r = min(static_cast<int>(sqrt(max_norm * eps) * B), r_thresh);
    if (tid < B) {
        // U的第0列 = 主列
        d_U[tid * r] = shared_A[tid][pivot_col];
        // V^T的第0行 = 主列
        d_Vt[pivot_col] = shared_A[tid][pivot_col];
    }
    if (tid == 0) {
        *d_r = r;
    }
}

// GPU端超级块划分核（论文4.2节Algorithm 1）
__global__ void block_partition_kernel(
    const float* d_A_csr_val,
    const int* d_A_csr_row_ptr,
    const int* d_A_csr_col_idx,
    size_t M,
    size_t K,
    int B,
    int r_thresh,
    float eps,
    char* d_block_type,
    int* d_block_p,
    int* d_block_q,
    float** d_dense_blocks,
    float** d_U_list,
    float** d_Vt_list,
    int* d_r_list
) {
    const size_t block_idx = blockIdx.x;
    const int p = d_block_p[block_idx];
    const int q = d_block_q[block_idx];
    const int row_start = p * B;
    const int row_end = min((p + 1) * B, static_cast<int>(M));
    const int col_start = q * B;
    const int col_end = min((q + 1) * B, static_cast<int>(K));

    // 步骤1：加载超级块数据至共享内存
    __shared__ float shared_A[64][64];
    __shared__ int shared_non_zero_cnt;
    if (threadIdx.x == 0) shared_non_zero_cnt = 0;
    __syncthreads();

    // 线程分工：每个线程处理1行
    if (threadIdx.x < row_end - row_start) {
        const int global_row = row_start + threadIdx.x;
        const int row_ptr_start = d_A_csr_row_ptr[global_row];
        const int row_ptr_end = d_A_csr_row_ptr[global_row + 1];

        // 初始化当前行
        for (int y = 0; y < B; y++) {
            shared_A[threadIdx.x][y] = 0.0f;
        }

        // 填充非零元素
        int non_zero = 0;
        for (int j = row_ptr_start; j < row_ptr_end; j++) {
            const int global_col = d_A_csr_col_idx[j];
            if (global_col >= col_start && global_col < col_end) {
                const int local_col = global_col - col_start;
                shared_A[threadIdx.x][local_col] = d_A_csr_val[j];
                non_zero++;
            }
        }

        // 原子计数非零元素
        atomicAdd(&shared_non_zero_cnt, non_zero);
    }
    __syncthreads();

    // 步骤2：计算非零率并分类
    const float non_zero_ratio = static_cast<float>(shared_non_zero_cnt) / (B * B);
    if (threadIdx.x == 0) {
        if (non_zero_ratio < 0.01) {
            d_block_type[block_idx] = 0;  // 零块
        } else if (non_zero_ratio > 0.5) {
            d_block_type[block_idx] = 2;  // 稠密块
            // 分配稠密块内存并拷贝数据
            float* d_dense = nullptr;
            cudaMalloc(&d_dense, B * B * sizeof(float));
            cudaMemcpy(d_dense, shared_A, B * B * sizeof(float), cudaMemcpySharedToDevice);
            d_dense_blocks[block_idx] = d_dense;
        } else {
            d_block_type[block_idx] = 1;  // 低秩块
            // 分配U/V^T内存
            float* d_U = nullptr;
            float* d_Vt = nullptr;
            cudaMalloc(&d_U, B * r_thresh * sizeof(float));
            cudaMalloc(&d_Vt, r_thresh * B * sizeof(float));
            d_U_list[block_idx] = d_U;
            d_Vt_list[block_idx] = d_Vt;

            // 交叉近似计算基向量
            int* d_r = nullptr;
            cudaMalloc(&d_r, sizeof(int));
            cross_approximation_kernel<<<1, B>>>(
                reinterpret_cast<float*>(shared_A), B, eps, r_thresh, d_U, d_Vt, d_r
            );
            cudaMemcpy(&d_r_list[block_idx], d_r, sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(d_r);
        }
    }
}

cudaError_t blr_sparse_preprocess(
    cusparseHandle_t cusparse_h,
    const HostCSRMatrix& host_csr,
    int r_thresh,
    int super_block_size,
    int sub_block_size,
    float zfp_eps,
    BLRMatrix& blr_A
) {
    // 步骤1：初始化BLRMatrix元数据
    const size_t M = host_csr.rows;
    const size_t K = host_csr.cols;
    const size_t block_cnt_p = (M + super_block_size - 1) / super_block_size;
    const size_t block_cnt_q = (K + super_block_size - 1) / super_block_size;
    blr_A.total_blocks = block_cnt_p * block_cnt_q;
    blr_A.mat_rows = M;
    blr_A.mat_cols = K;
    blr_A.super_block_size = super_block_size;
    blr_A.sub_block_size = sub_block_size;

    // 步骤2：分配元数据内存（GPU端）
    blr_A.block_type.resize(blr_A.total_blocks, 0);
    blr_A.block_p.resize(blr_A.total_blocks, 0);
    blr_A.block_q.resize(blr_A.total_blocks, 0);
    blr_A.r_list.resize(blr_A.total_blocks, 0);
    blr_A.subblock_cnt.resize(blr_A.total_blocks, 0);
    blr_A.morton_size.resize(blr_A.total_blocks, 0);

    // 填充块索引（p, q）
    for (size_t b = 0; b < blr_A.total_blocks; b++) {
        blr_A.block_p[b] = static_cast<int>(b / block_cnt_q);
        blr_A.block_q[b] = static_cast<int>(b % block_cnt_q);
    }

    // 步骤3：拷贝CSR矩阵至GPU
    float* d_A_val;
    int* d_A_row_ptr;
    int* d_A_col_idx;
    cudaMalloc(&d_A_val, host_csr.val.size() * sizeof(float));
    cudaMalloc(&d_A_row_ptr, host_csr.row_ptr.size() * sizeof(int));
    cudaMalloc(&d_A_col_idx, host_csr.col_idx.size() * sizeof(int));
    cudaMemcpy(d_A_val, host_csr.val.data(), host_csr.val.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_row_ptr, host_csr.row_ptr.data(), host_csr.row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_col_idx, host_csr.col_idx.data(), host_csr.col_idx.size() * sizeof(int), cudaMemcpyHostToDevice);

    // 步骤4：GPU并行块划分
    std::vector<float*> d_dense_blocks(blr_A.total_blocks, nullptr);
    std::vector<float*> d_U_list(blr_A.total_blocks, nullptr);
    std::vector<float*> d_Vt_list(blr_A.total_blocks, nullptr);
    std::vector<int> r_list(blr_A.total_blocks, 0);

    dim3 grid(blr_A.total_blocks), block(super_block_size);  // 每个线程块处理1个超级块
    block_partition_kernel<<<grid, block>>>(
        d_A_val, d_A_row_ptr, d_A_col_idx, M, K, super_block_size, r_thresh, zfp_eps,
        blr_A.block_type.data(), blr_A.block_p.data(), blr_A.block_q.data(),
        d_dense_blocks.data(), d_U_list.data(), d_Vt_list.data(), r_list.data()
    );
    CUDA_CHECK(cudaGetLastError());

    // 步骤5：稠密块Morton编码
    std::vector<float*> d_morton_blocks;
    std::vector<size_t> morton_sizes;
    std::vector<float*> dense_blocks_filtered;
    for (size_t b = 0; b < blr_A.total_blocks; b++) {
        if (blr_A.block_type[b] == 2) {
            dense_blocks_filtered.push_back(d_dense_blocks[b]);
        }
    }
    CUDA_CHECK(utils::morton_encode_batch_gpu(
        dense_blocks_filtered, dense_blocks_filtered.size(), sub_block_size,
        d_morton_blocks, morton_sizes
    ));

    // 步骤6：低秩块ZFP压缩
    std::vector<float*> U_filtered, Vt_filtered;
    std::vector<int> r_filtered;
    for (size_t b = 0; b < blr_A.total_blocks; b++) {
        if (blr_A.block_type[b] == 1) {
            U_filtered.push_back(d_U_list[b]);
            Vt_filtered.push_back(d_Vt_list[b]);
            r_filtered.push_back(r_list[b]);
        }
    }
    CUDA_CHECK(utils::zfp_gpu_batch_compress_lowrank(
        U_filtered, Vt_filtered, r_filtered, super_block_size, zfp_eps,
        blr_A.d_U_compressed, blr_A.d_Vt_compressed, blr_A.comp_len_U, blr_A.comp_len_Vt
    ));

    // 步骤7：整理BLRMatrix结构
    size_t dense_idx = 0, lowrank_idx = 0;
    for (size_t b = 0; b < blr_A.total_blocks; b++) {
        if (blr_A.block_type[b] == 1) {
            blr_A.r_list[b] = r_filtered[lowrank_idx++];
        } else if (blr_A.block_type[b] == 2) {
            blr_A.dense_morton.push_back(d_morton_blocks[dense_idx]);
            blr_A.subblock_cnt[b] = (super_block_size / sub_block_size) * (super_block_size / sub_block_size);
            blr_A.morton_size[b] = morton_sizes[dense_idx++];
        }
    }

    // 步骤8：释放临时内存
    cudaFree(d_A_val);
    cudaFree(d_A_row_ptr);
    cudaFree(d_A_col_idx);
    for (auto d : d_dense_blocks) if (d) cudaFree(d);
    for (auto u : d_U_list) if (u) cudaFree(u);
    for (auto vt : d_Vt_list) if (vt) cudaFree(vt);

    return cudaSuccess;
}

cudaError_t blr_decompress_lowrank_blocks(
    const BLRMatrix& blr_A,
    float zfp_eps,
    std::vector<float*>& d_U_decomp,
    std::vector<float*>& d_Vt_decomp
) {
    d_U_decomp.resize(blr_A.total_blocks, nullptr);
    d_Vt_decomp.resize(blr_A.total_blocks, nullptr);

    size_t lowrank_idx = 0;
    for (size_t b = 0; b < blr_A.total_blocks; b++) {
        if (blr_A.block_type[b] != 1) continue;

        const int r = blr_A.r_list[b];
        const size_t u_len = blr_A.super_block_size * r;
        const size_t vt_len = r * blr_A.super_block_size;

        // 解压缩U
        CUDA_CHECK(utils::zfp_gpu_decompress(
            blr_A.d_U_compressed[lowrank_idx], blr_A.comp_len_U[lowrank_idx],
            zfp_eps, u_len, d_U_decomp[b]
        ));

        // 解压缩V^T
        CUDA_CHECK(utils::zfp_gpu_decompress(
            blr_A.d_Vt_compressed[lowrank_idx], blr_A.comp_len_Vt[lowrank_idx],
            zfp_eps, vt_len, d_Vt_decomp[b]
        ));

        lowrank_idx++;
    }
    return cudaSuccess;
}

void blr_free_decompressed_blocks(
    std::vector<float*>& d_U_decomp,
    std::vector<float*>& d_Vt_decomp
) {
    for (auto u : d_U_decomp) if (u) cudaFree(u);
    for (auto vt : d_Vt_decomp) if (vt) cudaFree(vt);
    d_U_decomp.clear();
    d_Vt_decomp.clear();
}

bool blr_load_csr_from_bin(
    const std::string& val_path,
    const std::string& row_ptr_path,
    const std::string& col_idx_path,
    HostCSRMatrix& host_csr
) {
    // 读取非零元素值（float）
    std::ifstream val_fin(val_path, std::ios::binary);
    if (!val_fin.is_open()) return false;
    val_fin.seekg(0, std::ios::end);
    const size_t val_size = val_fin.tellg() / sizeof(float);
    host_csr.val.resize(val_size);
    val_fin.seekg(0, std::ios::beg);
    val_fin.read(reinterpret_cast<char*>(host_csr.val.data()), val_size * sizeof(float));
    val_fin.close();

    // 读取行指针（int）
    std::ifstream row_ptr_fin(row_ptr_path, std::ios::binary);
    if (!row_ptr_fin.is_open()) return false;
    row_ptr_fin.seekg(0, std::ios::end);
    const size_t row_ptr_size = row_ptr_fin.tellg() / sizeof(int);
    host_csr.row_ptr.resize(row_ptr_size);
    row_ptr_fin.seekg(0, std::ios::beg);
    row_ptr_fin.read(reinterpret_cast<char*>(host_csr.row_ptr.data()), row_ptr_size * sizeof(int));
    row_ptr_fin.close();

    // 读取列索引（int）
    std::ifstream col_idx_fin(col_idx_path, std::ios::binary);
    if (!col_idx_fin.is_open()) return false;
    col_idx_fin.seekg(0, std::ios::end);
    const size_t col_idx_size = col_idx_fin.tellg() / sizeof(int);
    host_csr.col_idx.resize(col_idx_size);
    col_idx_fin.seekg(0, std::ios::beg);
    col_idx_fin.read(reinterpret_cast<char*>(host_csr.col_idx.data()), col_idx_size * sizeof(int));
    col_idx_fin.close();

    // 计算矩阵维度
    host_csr.rows = host_csr.row_ptr.size() - 1;
    host_csr.nnz = host_csr.val.size();
    host_csr.cols = 0;
    for (int col : host_csr.col_idx) {
        if (col >= static_cast<int>(host_csr.cols)) {
            host_csr.cols = col + 1;
        }
    }

    return true;
}