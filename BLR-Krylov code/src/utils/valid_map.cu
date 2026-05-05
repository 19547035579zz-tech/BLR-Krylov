#include "utils/valid_map.cuh"
#include <cuda_runtime.h>
#include <cassert>

namespace utils {

__global__ void valid_map_init_kernel(bool* d_valid, size_t k, bool init_valid) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < k) {
        d_valid[tid] = init_valid;
    }
}

cudaError_t valid_map_init(
    size_t k,
    bool*& d_valid,
    bool init_valid
) {
    if (k == 0) return cudaErrorInvalidValue;

    cudaMalloc(&d_valid, k * sizeof(bool));
    if (!d_valid) return cudaErrorMemoryAllocation;

    dim3 grid((k + 255) / 256), block(256);
    valid_map_init_kernel<<<grid, block>>>(d_valid, k, init_valid);
    return cudaGetLastError();
}

__global__ void valid_map_update_kernel(
    bool* d_valid,
    const int* d_invalid_indices,
    size_t invalid_cnt
) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < invalid_cnt) {
        const int idx = d_invalid_indices[tid];
        d_valid[idx] = false;
    }
}

cudaError_t valid_map_batch_update(
    bool* d_valid,
    const std::vector<int>& invalid_indices,
    size_t invalid_cnt
) {
    if (!d_valid || invalid_cnt == 0) return cudaSuccess;

    int* d_invalid_indices;
    cudaMalloc(&d_invalid_indices, invalid_cnt * sizeof(int));
    cudaMemcpy(d_invalid_indices, invalid_indices.data(), invalid_cnt * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid((invalid_cnt + 255) / 256), block(256);
    valid_map_update_kernel<<<grid, block>>>(d_valid, d_invalid_indices, invalid_cnt);

    cudaFree(d_invalid_indices);
    return cudaGetLastError();
}

__global__ void valid_map_count_kernel(
    const bool* d_valid,
    size_t k,
    int* d_valid_cnt
) {
    __shared__ int shared_cnt[256];
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t lane = threadIdx.x;

    shared_cnt[lane] = 0;
    if (tid < k && d_valid[tid]) {
        shared_cnt[lane] = 1;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (lane < s) {
            shared_cnt[lane] += shared_cnt[lane + s];
        }
        __syncthreads();
    }

    if (lane == 0) {
        atomicAdd(d_valid_cnt, shared_cnt[0]);
    }
}

cudaError_t valid_map_query_count(
    const bool* d_valid,
    size_t k,
    int& valid_cnt
) {
    if (!d_valid || k == 0) {
        valid_cnt = 0;
        return cudaSuccess;
    }

    int* d_valid_cnt;
    cudaMalloc(&d_valid_cnt, sizeof(int));
    cudaMemset(d_valid_cnt, 0, sizeof(int));

    dim3 grid((k + 255) / 256), block(256);
    valid_map_count_kernel<<<grid, block>>>(d_valid, k, d_valid_cnt);

    cudaMemcpy(&valid_cnt, d_valid_cnt, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_valid_cnt);
    return cudaGetLastError();
}

__global__ void valid_map_filter_kernel(
    const float* d_V_in,
    const bool* d_valid,
    size_t M,
    size_t k,
    const int* d_valid_indices,
    int valid_cnt,
    float* d_V_out
) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M * valid_cnt) return;

    const size_t m = tid / valid_cnt;
    const size_t v_out_col = tid % valid_cnt;
    const size_t v_in_col = d_valid_indices[v_out_col];
    d_V_out[tid] = d_V_in[m * k + v_in_col];
}

cudaError_t valid_map_filter_basis(
    const float* d_V_in,
    const bool* d_valid,
    size_t M,
    size_t k,
    int valid_cnt,
    float*& d_V_out
) {
    if (!d_V_in || !d_valid || M == 0 || valid_cnt == 0) return cudaErrorInvalidDevicePointer;

    std::vector<int> valid_indices(valid_cnt);
    int cnt = 0;
    for (size_t v = 0; v < k && cnt < valid_cnt; v++) {
        bool valid;
        cudaMemcpy(&valid, &d_valid[v], sizeof(bool), cudaMemcpyDeviceToHost);
        if (valid) {
            valid_indices[cnt++] = static_cast<int>(v);
        }
    }

    cudaMalloc(&d_V_out, M * valid_cnt * sizeof(float));
    if (!d_V_out) return cudaErrorMemoryAllocation;

    int* d_valid_indices;
    cudaMalloc(&d_valid_indices, valid_cnt * sizeof(int));
    cudaMemcpy(d_valid_indices, valid_indices.data(), valid_cnt * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid((M * valid_cnt + 255) / 256), block(256);
    valid_map_filter_kernel<<<grid, block>>>(
        d_V_in, d_valid, M, k, d_valid_indices, valid_cnt, d_V_out
    );

    cudaFree(d_valid_indices);
    return cudaGetLastError();
}
}  // namespace utils
