#pragma once
#include "blr_krylov_types.h"
#include <cublas_v2.h>

/**
 * @brief 计算迭代残差（论文4.10节）
 * res = ||b - A×x||₂，GPU端完成，无主机端数据传输
 * @param cublas_h cuBLAS句柄
 * @param blr_A GPU端BLR-Sparse矩阵
 * @param d_U_decomp 解压缩后U基向量列表（GPU端）
 * @param d_Vt_decomp 解压缩后V^T基向量列表（GPU端）
 * @param d_x 当前解向量（M×1，GPU端float）
 * @param d_b 线性方程组右边项（M×1，GPU端float）
 * @param res [输出] 残差数值
 * @return cudaError_t 错误码
 */
cudaError_t iter_compute_residual(
    cublasHandle_t cublas_h,
    const BLRMatrix& blr_A,
    const std::vector<float*>& d_U_decomp,
    const std::vector<float*>& d_Vt_decomp,
    const float* d_x,
    const float* d_b,
    float& res
);

/**
 * @brief 动态调整低秩阈值（论文4.10节）
 * 根据当前残差调整r_thresh，平衡精度与性能
 * @param curr_res 当前残差
 * @param params 迭代控制参数
 * @param curr_r [输入/输出] 当前低秩阈值，输出调整后的值
 */
void iter_adjust_r_threshold(
    float curr_res,
    const IterParams& params,
    int& curr_r
);

/**
 * @brief 迭代主循环（论文4.10节）
 * 串联基向量计算、SpGEMM、残差监控、参数调整
 * @param cublas_h cuBLAS句柄
 * @param cusparse_h cuSPARSE句柄
 * @param blr_A GPU端BLR-Sparse矩阵
 * @param d_U_decomp 解压缩后U基向量列表（GPU端）
 * @param d_Vt_decomp 解压缩后V^T基向量列表（GPU端）
 * @param d_V0 初始基向量V0（GPU端float）
 * @param d_b 线性方程组右边项（M×1，GPU端float）
 * @param d_x [输入/输出] 解向量，输出最终结果（GPU端float）
 * @param params 迭代控制参数
 * @param k 基向量数量
 * @return cudaError_t 错误码
 */
cudaError_t iter_main_loop(
    cublasHandle_t cublas_h,
    cusparseHandle_t cusparse_h,
    const BLRMatrix& blr_A,
    const std::vector<float*>& d_U_decomp,
    const std::vector<float*>& d_Vt_decomp,
    const float* d_V0,
    const float* d_b,
    float* d_x,
    const IterParams& params,
    size_t k
);

/**
 * @brief 初始化迭代参数
 * @param res_threshold 残差阈值
 * @param r_init 初始低秩阈值
 * @param r_max 最大低秩阈值
 * @param r_min 最小低秩阈值
 * @param max_iter 最大迭代步数
 * @param zfp_eps ZFP压缩精度
 * @return IterParams 初始化后的参数
 */
IterParams iter_init_params(
    float res_threshold = 1e-6,
    int r_init = 32,
    int r_max = 64,
    int r_min = 16,
    int max_iter = 100,
    float zfp_eps = 1e-6
);