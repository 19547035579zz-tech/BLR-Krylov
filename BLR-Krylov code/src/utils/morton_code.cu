#include "utils/morton_code.cuh"
#include <cuda_runtime.h>
#include <cassert>

namespace utils {
// 2D坐标转Morton编码（按位交错，适用于2的幂尺寸）
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

// GPU端16×16子块Morton编码核
__global__ void morton_encode_16x16_kernel(
    const float* d_dense_block,
    float* d_morton_block
) {
    const int tid = threadIdx.x;
    if (tid >= 16 * 16) return;

    // 子块内坐标（x：行，y：列）
    const int x = tid / 16;
    const int y = tid % 16;
    // Morton编码（16是2^4，支持位交错）
    const size_t morton_code = morton_encode_bitwise(x, y, 16);
    // 写入Morton地址
    d_morton_block[morton_code] = d_dense_block[tid];
}

cudaError_t morton_encode_gpu_16x16(
    const float* d_dense_block,
    float* d_morton_block
) {
    if (!d_dense_block || !d_morton_block) return cudaErrorInvalidDevicePointer;

    // 16×16子块，1个线程块（256线程）足够
    dim3 grid(1), block(256);
    morton_encode_16x16_kernel<<<grid, block>>>(d_dense_block, d_morton_block);
    return cudaGetLastError();
}

// 批量编码稠密超级块（64×64→4×4个16×16子块）
cudaError_t morton_encode_batch_gpu(
    const std::vector<float*>& d_dense_blocks,
    size_t block_cnt,
    int sub_block_size,
    std::vector<float*>& d_morton_blocks,
    std::vector<size_t>& morton_sizes
) {
    assert(sub_block_size == 16 && "Only 16×16 sub-block supported");
    const size_t sub_block_cnt_per_super = (64 / 16) * (64 / 16) = 16;  // 64×64超级块含16个子块
    const size_t sub_block_size_bytes = sub_block_size * sub_block_size * sizeof(float);

    d_morton_blocks.resize(block_cnt, nullptr);
    morton_sizes.resize(block_cnt, sub_block_cnt_per_super * sub_block_size_bytes);

    // 为每个超级块分配Morton内存
    for (size_t b = 0; b < block_cnt; b++) {
        cudaMalloc(&d_morton_blocks[b], morton_sizes[b]);
        if (!d_morton_blocks[b]) return cudaErrorMemoryAllocation;

        // 拆分超级块为16×16子块并编码
        for (int sb_p = 0; sb_p < 4; sb_p++) {  // 行方向4个子块
            for (int sb_q = 0; sb_q < 4; sb_q++) {  // 列方向4个子块
                // 子块在超级块中的偏移
                const size_t sb_offset = (sb_p * 16 * 64) + (sb_q * 16);
                const float* d_sub_block = &d_dense_blocks[b][sb_offset];
                // 子块在Morton块中的偏移
                const size_t morton_sb_offset = (sb_p * 4 + sb_q) * 16 * 16;
                float* d_morton_sub = &d_morton_blocks[b][morton_sb_offset];

                // 编码单个子块
                CUDA_CHECK(morton_encode_gpu_16x16(d_sub_block, d_morton_sub));
            }
        }
    }
    return cudaSuccess;
}
}  // namespace utils