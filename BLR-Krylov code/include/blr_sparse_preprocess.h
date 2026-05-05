#pragma once
#include "blr_krylov_types.h"
#include "utils/zfp_gpu.cuh"
#include "utils/morton_code.cuh"
#include <cusparse.h>

cudaError_t blr_sparse_preprocess(
    cusparseHandle_t cusparse_h,
    const HostCSRMatrix& host_csr,
    int r_thresh,
    int super_block_size,
    int sub_block_size,
    float zfp_eps,
    BLRMatrix& blr_A
);

cudaError_t blr_decompress_lowrank_blocks(
    const BLRMatrix& blr_A,
    float zfp_eps,
    std::vector<float*>& d_U_decomp,
    std::vector<float*>& d_Vt_decomp
);

void blr_free_decompressed_blocks(
    std::vector<float*>& d_U_decomp,
    std::vector<float*>& d_Vt_decomp
);

bool blr_load_csr_from_bin(
    const std::string& val_path,
    const std::string& row_ptr_path,
    const std::string& col_idx_path,
    HostCSRMatrix& host_csr
);
