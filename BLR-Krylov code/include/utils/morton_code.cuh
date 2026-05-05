#pragma once
#include <cuda_runtime.h>
#include <vector>

namespace utils {
/**
 * @brief Convert 2D coordinates to Morton code (Host side, for validation)
 * @param x Row coordinate (0 to size-1)
 * @param y Column coordinate (0 to size-1)
 * @param size Dimension size (must be a power of 2)
 * @return size_t Morton code value
 */
size_t morton_encode_host(int x, int y, int size);

/**
 * @brief Convert Morton code to 2D coordinates (Host side, for validation)
 * @param code Morton code value
 * @param size Dimension size (must be a power of 2)
 * @param x [out] Row coordinate
 * @param y [out] Column coordinate
 */
void morton_decode_host(size_t code, int size, int& x, int& y);

/**
 * @brief Z-Morton encoding on GPU (16x16 sub-block)
 * @param d_dense_block Input dense block (16x16, float on GPU)
 * @param d_morton_block [out] Encoded Morton data (float on GPU)
 * @return cudaError_t Error code
 */
cudaError_t morton_encode_gpu_16x16(
    const float* d_dense_block,
    float* d_morton_block
);

/**
 * @brief Batched encoding of dense super-blocks (on GPU)
 * @param d_dense_blocks List of dense super-blocks (64x64, float on GPU)
 * @param block_cnt Number of super-blocks
 * @param sub_block_size Sub-block size (default 16)
 * @param d_morton_blocks [out] List of encoded Morton blocks (on GPU)
 * @param morton_sizes [out] Size of each Morton block (in bytes)
 * @return cudaError_t Error code
 */
cudaError_t morton_encode_batch_gpu(
    const std::vector<float*>& d_dense_blocks,
    size_t block_cnt,
    int sub_block_size,
    std::vector<float*>& d_morton_blocks,
    std::vector<size_t>& morton_sizes
);
}  // namespace utils
