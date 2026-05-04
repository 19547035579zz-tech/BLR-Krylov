#pragma once
#include "blr_krylov_types.h"

/**
 * @brief 低秩块SpGEMM（基向量内积，论文4.6节）
 * 计算C_pq = U_pq × (V^T_pq × V)，GPU端并行执行
 * @param d_U 低秩块U基向量（B×r，GPU端float）
 * @param d_Vt 低秩块V^T基向量（r×B，GPU端float）
 * @param d_V 输入基向量矩阵（B×k，GPU端float）
 * @param B 超级块尺寸
 * @param r 低秩块秩
 * @param k 基向量数量
 * @param d_C_block [输出] 低秩块计算结果（B×k，GPU端float）
 * @return cudaError_t 错误码
 */
cudaError_t fused_spgemm_lowrank_block(
    const float* d_U,
    const float* d_Vt,
    const float* d_V,
    int B,
    int r,
    size_t k,
    float* d_C_block
);

/**
 * @brief 稠密块SpGEMM（张量核加速，论文4.7节）
 * 调用NVIDIA mma.sync指令（FP16），计算C_pq = A_pq × V
 * @param d_A_morton 稠密块Morton编码（16×16子块，GPU端float）
 * @param d_V 输入基向量矩阵（B×k，GPU端float）
 * @param B 超级块尺寸
 * @param k 基向量数量
 * @param d_C_block [输出] 稠密块计算结果（B×k，GPU端float）
 * @return cudaError_t 错误码
 */
cudaError_t fused_spgemm_dense_block_tensor_core(
    const float* d_A_morton,
    const float* d_V,
    int B,
    size_t k,
    float* d_C_block
);

/**
 * @brief 融合SpGEMM执行（论文4.6-4.8节）
 * 批量处理所有超级块，差异化计算低秩/稠密/零块
 * @param blr_A GPU端BLR-Sparse矩阵
 * @param d_U_decomp 解压缩后U基向量列表（GPU端）
 * @param d_Vt_decomp 解压缩后V^T基向量列表（GPU端）
 * @param basis 批量基向量（GPU端）
 * @param d_C [输出] SpGEMM结果C=A×V（M×k，GPU端float）
 * @return cudaError_t 错误码
 */
cudaError_t fused_spgemm_execute(
    const BLRMatrix& blr_A,
    const std::vector<float*>& d_U_decomp,
    const std::vector<float*>& d_Vt_decomp,
    const BasisVectors& basis,
    float* d_C
);

/**
 * @brief 块结果聚合（论文4.8节）
 * 线程块缓存+全局合并写，减少全局内存写次数
 * @param d_C_blocks 各超级块计算结果（GPU端，按块存储）
 * @param blr_A GPU端BLR-Sparse矩阵
 * @param k 基向量数量
 * @param d_C [输出] 聚合后的完整结果（M×k，GPU端float）
 * @return cudaError_t 错误码
 */
cudaError_t fused_spgemm_aggregate(
    const std::vector<float*>& d_C_blocks,
    const BLRMatrix& blr_A,
    size_t k,
    float* d_C
);