#pragma once
#include <cuda_runtime.h>
#include <vector>

namespace utils {
/**
 * @brief 初始化ValidMap（基向量有效性标记）
 * @param k 基向量数量
 * @param d_valid [输出] 有效性标记（GPU端bool）
 * @param init_valid 是否初始化为有效（true=全部有效）
 * @return cudaError_t 错误码
 */
cudaError_t valid_map_init(
    size_t k,
    bool*& d_valid,
    bool init_valid = true
);

/**
 * @brief 批量更新ValidMap（标记无效基向量）
 * @param d_valid 有效性标记（GPU端bool）
 * @param invalid_indices 无效基向量索引列表（主机端）
 * @param invalid_cnt 无效基向量数量
 * @return cudaError_t 错误码
 */
cudaError_t valid_map_batch_update(
    bool* d_valid,
    const std::vector<int>& invalid_indices,
    size_t invalid_cnt
);

/**
 * @brief 查询有效基向量数量
 * @param d_valid 有效性标记（GPU端bool）
 * @param k 基向量总数
 * @param valid_cnt [输出] 有效基向量数量
 * @return cudaError_t 错误码
 */
cudaError_t valid_map_query_count(
    const bool* d_valid,
    size_t k,
    int& valid_cnt
);

/**
 * @brief 筛选有效基向量（提取有效向量至新矩阵）
 * @param d_V_in 输入基向量矩阵（M×k，GPU端float）
 * @param d_valid 有效性标记（GPU端bool）
 * @param M 基向量行数
 * @param k 基向量总数
 * @param valid_cnt 有效基向量数量
 * @param d_V_out [输出] 有效基向量矩阵（M×valid_cnt，GPU端）
 * @return cudaError_t 错误码
 */
cudaError_t valid_map_filter_basis(
    const float* d_V_in,
    const bool* d_valid,
    size_t M,
    size_t k,
    int valid_cnt,
    float*& d_V_out
);
}  // namespace utils