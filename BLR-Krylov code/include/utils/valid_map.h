#pragma once
#include <cuda_runtime.h>
#include <vector>

namespace utils {
/**
 * @brief Initialize ValidMap (basis vector validity flags)
 * @param k Number of basis vectors
 * @param d_valid [out] Validity flags (bool on GPU)
 * @param init_valid Whether to initialize as valid (true = all valid)
 * @return cudaError_t Error code
 */
cudaError_t valid_map_init(
    size_t k,
    bool*& d_valid,
    bool init_valid = true
);

/**
 * @brief Batch update ValidMap (mark invalid basis vectors)
 * @param d_valid Validity flags (bool on GPU)
 * @param invalid_indices List of invalid basis vector indices (Host side)
 * @param invalid_cnt Number of invalid basis vectors
 * @return cudaError_t Error code
 */
cudaError_t valid_map_batch_update(
    bool* d_valid,
    const std::vector<int>& invalid_indices,
    size_t invalid_cnt
);

/**
 * @brief Query the number of valid basis vectors
 * @param d_valid Validity flags (bool on GPU)
 * @param k Total number of basis vectors
 * @param valid_cnt [out] Number of valid basis vectors
 * @return cudaError_t Error code
 */
cudaError_t valid_map_query_count(
    const bool* d_valid,
    size_t k,
    int& valid_cnt
);

/**
 * @brief Filter valid basis vectors (extract valid vectors to a new matrix)
 * @param d_V_in Input basis vector matrix (M x k, float on GPU)
 * @param d_valid Validity flags (bool on GPU)
 * @param M Number of rows in the basis vectors
 * @param k Total number of basis vectors
 * @param valid_cnt Number of valid basis vectors
 * @param d_V_out [out] Matrix of valid basis vectors (M x valid_cnt, on GPU)
 * @return cudaError_t Error code
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
