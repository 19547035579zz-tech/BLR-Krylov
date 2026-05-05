#include "fused_spgemm.h"
#include <cuda_runtime.h>
#include <mma.h>
#include <cassert>

__global__ void lowrank_block_gemm_kernel(
    const float* d_U,
    const float* d_Vt,
    const float* d_V,
    int B,
    int r,
    size_t k,
    float* d_C_block
) {
    const int tid = threadIdx.x;
    __shared__ float sm_VtV[r_max][k_max];
    __shared__ float sm_U[r_max][B];

    if (tid < r * B) {
        const int rt = tid / B;
        const int bt = tid % B;
        sm_U[rt][bt] = d_U[bt * r + rt];
    }
    __syncthreads();

    if (tid < r * k) {
        const int rt = tid / k;
        const int kt = tid % k;
        float sum = 0.0f;
        for (int bt = 0; bt < B; bt++) {
            sum += d_Vt[rt * B + bt] * d_V[bt * k + kt];
        }
        sm_VtV[rt][kt] = sum;
    }
    __syncthreads();

    if (tid < B * k) {
        const int bt = tid / k;
        const int kt = tid % k;
        float sum = 0.0f;
        for (int rt = 0; rt < r; rt++) {
            sum += sm_U[rt][bt] * sm_VtV[rt][kt];
        }
        d_C_block[tid] = sum;
    }
}

using namespace nvcuda;
__global__ void dense_block_tensor_core_kernel(
    const float* d_A_morton,
    const float* d_V,
    int B,
    size_t k,
    float* d_C_block
) {
    using mma_t = typename mma::experimental::mma_sync<float, 16, 8, 8, mma::experimental::layout::row_major>;
    mma_t::fragment_a a_frag;
    mma_t::fragment_b b_frag;
    mma_t::fragment_c c_frag;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int sub_block_cnt = (B / 16) * (B / 16);
    const int sub_block_idx = warp_id % sub_block_cnt;
    const int sb_p = (sub_block_idx / 4) * 16;
    const int sb_q = (sub_block_idx % 4) * 16;

    mma::experimental::fill_fragment(c_frag, 0.0f);

    for (size_t kt = 0; kt < k; kt++) {
        mma::experimental::load_matrix_sync(a_frag, 
            &d_A_morton[sb_p * B + sb_q], 
            B);

        mma::experimental::load_matrix_sync(b_frag, 
            &d_V[sb_q * k + kt], 
            k);

        mma::experimental::mma_sync(c_frag, a_frag, b_frag, c_frag);

        mma::experimental::store_matrix_sync(
            &d_C_block[(sb_p * k) + kt], 
            c_frag,
            k, 
            mma::experimental::layout::row_major);
    }
}

cudaError_t fused_spgemm_lowrank_block(
    const float* d_U,
    const float* d_Vt,
    const float* d_V,
    int B,
    int r,
    size_t k,
    float* d_C_block
) {
    if (!d_U || !d_Vt || !d_V || !d_C_block) return cudaErrorInvalidDevicePointer;
    if (B != 64 || r > 32) return cudaErrorInvalidValue;

    const dim3 grid(1);
    const dim3 block(max({r * B, r * k, static_cast<int>(B * k)}));
    lowrank_block_gemm_kernel<<<grid, block>>>(d_U, d_Vt, d_V, B, r, k, d_C_block);
    return cudaGetLastError();
}

cudaError_t fused_spgemm_dense_block_tensor_core(
    const float* d_A_morton,
    const float* d_V,
    int B,
    size_t k,
    float* d_C_block
) {
    if (!d_A_morton || !d_V || !d_C_block) return cudaErrorInvalidDevicePointer;
    if (B != 64 || k > 8) return cudaErrorInvalidValue;

    const dim3 grid(1);
    const dim3 block(256);
    dense_block_tensor_core_kernel<<<grid, block>>>(d_A_morton, d_V, B, k, d_C_block);
    return cudaGetLastError();
}

cudaError_t fused_spgemm_execute(
    const BLRMatrix& blr_A,
    const std::vector<float*>& d_U_decomp,
    const std::vector<float*>& d_Vt_decomp,
    const BasisVectors& basis,
    float* d_C
) {
    const size_t M = blr_A.mat_rows;
    const int B = blr_A.super_block_size;
    const size_t k = basis.cols;

    std::vector<float*> d_C_blocks(blr_A.total_blocks, nullptr);
    for (size_t b = 0; b < blr_A.total_blocks; b++) {
        cudaMalloc(&d_C_blocks[b], B * k * sizeof(float));
        cudaMemset(d_C_blocks[b], 0, B * k * sizeof(float));
    }

    for (size_t b = 0; b < blr_A.total_blocks; b++) {
        const int p = blr_A.block_p[b];
        const int q = blr_A.block_q[b];
        const int col_start = q * B;
        const float* d_V_block = &basis.d_V[col_start * k];

        switch (blr_A.block_type[b]) {
            case 1:
                CUDA_CHECK(fused_spgemm_lowrank_block(
                    d_U_decomp[b], d_Vt_decomp[b], d_V_block,
                    B, blr_A.r_list[b], k, d_C_blocks[b]
                ));
                break;
            case 2:
                CUDA_CHECK(fused_spgemm_dense_block_tensor_core(
                    blr_A.dense_morton[b], d_V_block,
                    B, k, d_C_blocks[b]
                ));
                break;
            case 0:
                break;
            default:
                return cudaErrorInvalidValue;
        }
    }

    CUDA_CHECK(fused_spgemm_aggregate(d_C_blocks, blr_A, k, d_C));

    for (auto d : d_C_blocks) if (d) cudaFree(d);
    return cudaSuccess;
}

__global__ void spgemm_aggregate_kernel(
    const float** d_C_blocks,
    const int* d_block_p,
    const int* d_block_q,
    int B,
    size_t k,
    size_t total_blocks,
    float* d_C
) {
    const size_t block_idx = blockIdx.x;
    if (block_idx >= total_blocks) return;

    const int p = d_block_p[block_idx];
    const int row_start = p * B;
    const float* d_C_block = d_C_blocks[block_idx];

    const int tid = threadIdx.x;
    if (tid < B * k) {
        const int bt = tid / k;
        const int kt = tid % k;
        const int global_row = row_start + bt;
        d_C[global_row * k + kt] = d_C_block[tid];
    }
}

cudaError_t fused_spgemm_aggregate(
    const std::vector<float*>& d_C_blocks,
    const BLRMatrix& blr_A,
    size_t k,
    float* d_C
) {
    if (d_C_blocks.empty() || !d_C) return cudaErrorInvalidDevicePointer;

    float** d_C_blocks_dev;
    cudaMalloc(&d_C_blocks_dev, d_C_blocks.size() * sizeof(float*));
    cudaMemcpy(d_C_blocks_dev, d_C_blocks.data(), d_C_blocks.size() * sizeof(float*), cudaMemcpyHostToDevice);

    const dim3 grid(blr_A.total_blocks);
    const dim3 block(blr_A.super_block_size * k);
    spgemm_aggregate_kernel<<<grid, block>>>(
        d_C_blocks_dev, blr_A.block_p.data(), blr_A.block_q.data(),
        blr_A.super_block_size, k, blr_A.total_blocks, d_C
    );
    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_C_blocks_dev);
    return cudaSuccess;
}
