#include "utils/morton_code.cuh"
#include <cuda_runtime.h>
#include <cassert>

namespace utils {

__device__ __host__ size_t morton_encode_bitwise(int x, int y, int size) {
    size_t code = 0;
    int level = 0;
    while (size > 1) {
        size >>= 1;
        code |= ((x & size) << (2 * level + 1)) | ((y & size) << (2 * level));
        level++;
    }
    return code;
}

size_t morton_encode_host(int x, int y, int size) {
    assert((size & (size - 1)) == 0 && "Size must be power of 2");
    return morton_encode_bitwise(x, y, size);
}

void morton_decode_host(size_t code, int size, int& x, int& y) {
    assert((size & (size - 1)) == 0 && "Size must be power of 2");
    x = 0, y = 0;
    int level = 0;
    while (size > 1) {
        size >>= 1;
        x |= ((code >> (2 * level + 1)) & size);
        y |= ((code >> (2 * level)) & size);
        level++;
    }
}

__global__ void morton_encode_16x16_kernel(
    const float* d_dense_block,
    float* d_morton_block
) {
    const int tid = threadIdx.x;
    if (tid >= 16 * 16) return;

    const int x = tid / 16;
    const int y = tid % 16;
    const size_t morton_code = morton_encode_bitwise(x, y, 16);
    d_morton_block[morton_code] = d_dense_block[tid];
}

cudaError_t morton_encode_gpu_16x16(
    const float* d_dense_block,
    float* d_morton_block
) {
    if (!d_dense_block || !d_morton_block) return cudaErrorInvalidDevicePointer;

    dim3 grid(1), block(256);
    morton_encode_16x16_kernel<<<grid, block>>>(d_dense_block, d_morton_block);
    return cudaGetLastError();
}

cudaError_t morton_encode_batch_gpu(
    const std::vector<float*>& d_dense_blocks,
    size_t block_cnt,
    int sub_block_size,
    std::vector<float*>& d_morton_blocks,
    std::vector<size_t>& morton_sizes
) {
    assert(sub_block_size == 16 && "Only 16×16 sub-block supported");
    const size_t sub_block_cnt_per_super = (64 / 16) * (64 / 16);
    const size_t sub_block_size_bytes = sub_block_size * sub_block_size * sizeof(float);

    d_morton_blocks.resize(block_cnt, nullptr);
    morton_sizes.resize(block_cnt, sub_block_cnt_per_super * sub_block_size_bytes);

    for (size_t b = 0; b < block_cnt; b++) {
        cudaMalloc(&d_morton_blocks[b], morton_sizes[b]);
        if (!d_morton_blocks[b]) return cudaErrorMemoryAllocation;

        for (int sb_p = 0; sb_p < 4; sb_p++) {
            for (int sb_q = 0; sb_q < 4; sb_q++) {
                const size_t sb_offset = (sb_p * 16 * 64) + (sb_q * 16);
                const float* d_sub_block = &d_dense_blocks[b][sb_offset];
                const size_t morton_sb_offset = (sb_p * 4 + sb_q) * 16 * 16;
                float* d_morton_sub = &d_morton_blocks[b][morton_sb_offset];

                CUDA_CHECK(morton_encode_gpu_16x16(d_sub_block, d_morton_sub));
            }
        }
    }
    return cudaSuccess;
}
}  // namespace utils
