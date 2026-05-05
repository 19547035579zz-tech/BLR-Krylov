#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>

struct BLRMatrix {
    std::vector<float*> d_U_compressed;
    std::vector<float*> d_Vt_compressed;
    std::vector<size_t> comp_len_U;
    std::vector<size_t> comp_len_Vt;
    std::vector<int> r_list;

    std::vector<float*> d_dense_morton;
    std::vector<int> subblock_cnt;
    std::vector<size_t> morton_size;

    std::vector<char> block_type;
    std::vector<int> block_p;
    std::vector<int> block_q;
    size_t total_blocks;
    size_t mat_rows;
    size_t mat_cols;
    int super_block_size;
    int sub_block_size;

    void free() {
        for (auto u : d_U_compressed) if (u) cudaFree(u);
        for (auto vt : d_Vt_compressed) if (vt) cudaFree(vt);
        for (auto dense : d_dense_morton) if (dense) cudaFree(dense);
        d_U_compressed.clear();
        d_Vt_compressed.clear();
        d_dense_morton.clear();
    }
};

struct BasisVectors {
    float* d_V;
    bool* d_valid;
    size_t rows;
    size_t cols;
    int curr_valid_cnt;

    void free() {
        if (d_V) cudaFree(d_V);
        if (d_valid) cudaFree(d_valid);
        d_V = nullptr;
        d_valid = nullptr;
    }
};

struct IterParams {
    float res_threshold;
    int r_init;
    int r_max;
    int r_min;
    int max_iter;
    float zfp_eps;
};

struct HostCSRMatrix {
    std::vector<float> val;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    size_t rows;
    size_t cols;
    size_t nnz;
};
