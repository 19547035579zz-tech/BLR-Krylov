#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>

/**
 * @brief BLR-Sparse矩阵结构（论文4.2-4.3节）
 * 存储低秩块、稠密块、零块及元数据，适配GPU内存层次
 */
struct BLRMatrix {
    // 低秩块：ZFP压缩后的基向量（GPU端）
    std::vector<float*> d_U_compressed;   // U_pq: [块数] × (压缩后数据)
    std::vector<float*> d_Vt_compressed;  // V^T_pq: [块数] × (压缩后数据)
    std::vector<size_t> comp_len_U;       // 每个U块的压缩后长度（字节）
    std::vector<size_t> comp_len_Vt;      // 每个V^T块的压缩后长度（字节）
    std::vector<int> r_list;              // 每个低秩块的数值秩r（≤R_THRESH）

    // 稠密块：Z-Morton编码的16×16子块（GPU端）
    std::vector<float*> d_dense_morton;   // 稠密子块（Morton编码）
    std::vector<int> subblock_cnt;        // 每个稠密超级块的子块数量（4×4=16）
    std::vector<size_t> morton_size;      // 每个稠密块的Morton数据大小（字节）

    // 元数据（论文4.3节表2）
    std::vector<char> block_type;         // 块类型：0=零块，1=低秩块，2=稠密块
    std::vector<int> block_p;             // 超级块行索引（p）
    std::vector<int> block_q;             // 超级块列索引（q）
    size_t total_blocks;                  // 超级块总数
    size_t mat_rows;                      // 原始矩阵行数（M）
    size_t mat_cols;                      // 原始矩阵列数（K）
    int super_block_size;                 // 超级块尺寸（默认64）
    int sub_block_size;                   // 子块尺寸（默认16）

    /**
     * @brief 释放GPU内存
     */
    void free() {
        for (auto u : d_U_compressed) if (u) cudaFree(u);
        for (auto vt : d_Vt_compressed) if (vt) cudaFree(vt);
        for (auto dense : d_dense_morton) if (dense) cudaFree(dense);
        d_U_compressed.clear();
        d_Vt_compressed.clear();
        d_dense_morton.clear();
    }
};

/**
 * @brief 批量基向量结构（论文4.4-4.5节）
 * 包含基向量数据与有效性标记（ValidMap），支持迭代复用
 */
struct BasisVectors {
    float* d_V;          // 基向量矩阵（M×k，GPU端）：列优先存储V0~Vk-1
    bool* d_valid;       // 基向量有效性标记（k个bool，GPU端）：true=有效
    size_t rows;         // 基向量行数（=M）
    size_t cols;         // 基向量列数（=k）
    int curr_valid_cnt;  // 当前有效基向量数量

    /**
     * @brief 释放GPU内存
     */
    void free() {
        if (d_V) cudaFree(d_V);
        if (d_valid) cudaFree(d_valid);
        d_V = nullptr;
        d_valid = nullptr;
    }
};

/**
 * @brief 迭代控制参数（论文4.10节）
 * 控制残差阈值、低秩阈值范围、最大迭代步数
 */
struct IterParams {
    float res_threshold;  // 残差收敛阈值（默认1e-6）
    int r_init;           // 初始低秩阈值（默认32）
    int r_max;            // 最大低秩阈值（默认64）
    int r_min;            // 最小低秩阈值（默认16）
    int max_iter;         // 最大迭代步数（默认100）
    float zfp_eps;        // ZFP压缩精度（默认1e-6）
};

/**
 * @brief CSR矩阵结构（主机端）
 * 存储输入稀疏矩阵的CSR格式数据，用于加载至GPU
 */
struct HostCSRMatrix {
    std::vector<float> val;    // 非零元素值（float）
    std::vector<int> row_ptr;  // 行指针（int）
    std::vector<int> col_idx;  // 列索引（int）
    size_t rows;               // 矩阵行数
    size_t cols;               // 矩阵列数
    size_t nnz;                // 非零元素数
};