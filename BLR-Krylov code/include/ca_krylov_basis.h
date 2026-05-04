#pragma once
#include "blr_krylov_types.h"
#include "utils/valid_map.cuh"

/**
 * @brief 矩阵幂核：批量计算Krylov基向量（论文4.4节Algorithm 2）
 * @param blr_A GPU端BLR-Sparse矩阵
 * @param d_U_decomp 解压缩后U基向量列表（GPU端）
 * @param d_Vt_decomp 解压缩后V^T基向量列表（GPU端）
 * @param d_V0 初始基向量V0（M×1，GPU端float）
 * @param k 基向量数量
 * @param prev_basis 上一轮基向量（用于复用，可为nullptr）
 * @param curr_basis [输出] 当前轮批量基向量（GPU端）
 * @return cudaError_t 错误码
 */
cudaError_t ca_krylov_compute_basis(
    const BLRMatrix& blr_A,
    const std::vector<float*>& d_U_decomp,
    const std::vector<float*>& d_Vt_decomp,
    const float* d_V0,
    size_t k,
    const BasisVectors* prev_basis,
    BasisVectors& curr_basis
);

/**
 * @brief 基向量迭代复用（论文4.5节）
 * 根据残差标记无效基向量，保留有效向量用于下一轮计算
 * @param prev_basis 上一轮基向量
 * @param curr_res 当前轮残差
 * @param res_threshold 残差阈值
 * @param curr_basis [输出] 复用后的当前基向量
 * @return cudaError_t 错误码
 */
cudaError_t ca_krylov_reuse_basis(
    const BasisVectors& prev_basis,
    float curr_res,
    float res_threshold,
    BasisVectors& curr_basis
);

/**
 * @brief 释放基向量内存
 * @param basis 基向量结构
 */
void ca_krylov_free_basis(BasisVectors& basis);