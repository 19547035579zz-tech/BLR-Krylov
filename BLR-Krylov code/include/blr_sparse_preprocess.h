#pragma once
#include "blr_krylov_types.h"
#include "utils/zfp_gpu.cuh"
#include "utils/morton_code.cuh"
#include <cusparse.h>

/**
 * @brief BLR-Sparse预处理（论文4.2-4.3节）
 * 将CSR格式稀疏矩阵转换为GPU优化的BLR-Sparse格式
 * @param cusparse_h cuSPARSE句柄
 * @param host_csr 主机端CSR矩阵
 * @param r_thresh 低秩块阈值（r≤r_thresh为低秩块）
 * @param super_block_size 超级块尺寸（默认64）
 * @param sub_block_size 子块尺寸（默认16）
 * @param zfp_eps ZFP压缩精度
 * @param blr_A [输出] GPU端BLR-Sparse矩阵
 * @return cudaError_t 错误码
 */
cudaError_t blr_sparse_preprocess(
    cusparseHandle_t cusparse_h,
    const HostCSRMatrix& host_csr,
    int r_thresh,
    int super_block_size,
    int sub_block_size,
    float zfp_eps,
    BLRMatrix& blr_A
);

/**
 * @brief 低秩块基向量解压缩（论文4.3节）
 * 预处理后首次使用前调用，将压缩基向量解压缩为可计算格式
 * @param blr_A GPU端BLR-Sparse矩阵
 * @param zfp_eps ZFP压缩精度（需与压缩时一致）
 * @param d_U_decomp [输出] 解压缩后U基向量列表（GPU端）
 * @param d_Vt_decomp [输出] 解压缩后V^T基向量列表（GPU端）
 * @return cudaError_t 错误码
 */
cudaError_t blr_decompress_lowrank_blocks(
    const BLRMatrix& blr_A,
    float zfp_eps,
    std::vector<float*>& d_U_decomp,
    std::vector<float*>& d_Vt_decomp
);

/**
 * @brief 释放低秩块解压缩内存
 * @param d_U_decomp 解压缩后U基向量列表
 * @param d_Vt_decomp 解压缩后V^T基向量列表
 */
void blr_free_decompressed_blocks(
    std::vector<float*>& d_U_decomp,
    std::vector<float*>& d_Vt_decomp
);

/**
 * @brief 加载CSR矩阵从二进制文件（主机端）
 * @param val_path A_val.bin路径
 * @param row_ptr_path A_row_ptr.bin路径
 * @param col_idx_path A_col_idx.bin路径
 * @param host_csr [输出] 主机端CSR矩阵
 * @return bool 加载成功返回true
 */
bool blr_load_csr_from_bin(
    const std::string& val_path,
    const std::string& row_ptr_path,
    const std::string& col_idx_path,
    HostCSRMatrix& host_csr
);