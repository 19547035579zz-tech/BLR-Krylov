#pragma once
#include <cuda_runtime.h>
#include <vector>

namespace utils {
/**
 * @brief 2D坐标转Morton编码（主机端，用于验证）
 * @param x 行坐标（0~size-1）
 * @param y 列坐标（0~size-1）
 * @param size 维度大小（需为2的幂）
 * @return size_t Morton编码值
 */
size_t morton_encode_host(int x, int y, int size);

/**
 * @brief Morton编码转2D坐标（主机端，用于验证）
 * @param code Morton编码值
 * @param size 维度大小（需为2的幂）
 * @param x [输出] 行坐标
 * @param y [输出] 列坐标
 */
void morton_decode_host(size_t code, int size, int& x, int& y);

/**
 * @brief GPU端Z-Morton编码（16×16子块）
 * @param d_dense_block 输入稠密块（16×16，GPU端float）
 * @param d_morton_block [输出] 编码后Morton数据（GPU端float）
 * @return cudaError_t 错误码
 */
cudaError_t morton_encode_gpu_16x16(
    const float* d_dense_block,
    float* d_morton_block
);

/**
 * @brief 批量编码稠密超级块（GPU端）
 * @param d_dense_blocks 稠密超级块列表（64×64，GPU端float）
 * @param block_cnt 超级块数量
 * @param sub_block_size 子块尺寸（默认16）
 * @param d_morton_blocks [输出] 编码后Morton列表（GPU端）
 * @param morton_sizes [输出] 每个Morton块的大小（字节）
 * @return cudaError_t 错误码
 */
cudaError_t morton_encode_batch_gpu(
    const std::vector<float*>& d_dense_blocks,
    size_t block_cnt,
    int sub_block_size,
    std::vector<float*>& d_morton_blocks,
    std::vector<size_t>& morton_sizes
);
}  // namespace utils