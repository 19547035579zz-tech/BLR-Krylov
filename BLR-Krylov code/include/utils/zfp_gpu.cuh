#pragma once
#include <cuda_runtime.h>
#include <zfp.h>
#include <vector>

namespace utils {
/**
 * @brief ZFP compression on GPU (FP32)
 * @param d_input Input data (float on GPU)
 * @param input_len Length of input data (number of elements)
 * @param eps Compression precision (controls error tolerance)
 * @param d_output [out] Compressed data (float on GPU)
 * @param output_len [out] Length of compressed data (in bytes)
 * @return cudaError_t Error code
 */
cudaError_t zfp_gpu_compress(
    const float* d_input,
    size_t input_len,
    float eps,
    float*& d_output,
    size_t& output_len
);

/**
 * @brief ZFP decompression on GPU (FP32)
 * @param d_input Compressed data (float on GPU)
 * @param input_len Length of compressed data (in bytes)
 * @param eps Compression precision (must match the precision used for compression)
 * @param output_len Length of decompressed data (number of elements)
 * @param d_output [out] Decompressed data (float on GPU)
 * @return cudaError_t Error code
 */
cudaError_t zfp_gpu_decompress(
    const float* d_input,
    size_t input_len,
    float eps,
    size_t output_len,
    float*& d_output
);

/**
 * @brief Batched compression of low-rank block basis vectors (on GPU)
 * @param d_U_list List of U basis vectors for low-rank blocks (float on GPU)
 * @param d_Vt_list List of V^T basis vectors for low-rank blocks (float on GPU)
 * @param r_list The rank 'r' of each low-rank block
 * @param block_size Super-block size (default 64)
 * @param eps Compression precision
 * @param d_U_compressed [out] List of compressed U vectors (on GPU)
 * @param d_Vt_compressed [out] List of compressed V^T vectors (on GPU)
 * @param comp_len_U [out] Compressed lengths of U vectors (in bytes)
 * @param comp_len_Vt [out] Compressed lengths of V^T vectors (in bytes)
 * @return cudaError_t Error code
 */
cudaError_t zfp_gpu_batch_compress_lowrank(
    const std::vector<float*>& d_U_list,
    const std::vector<float*>& d_Vt_list,
    const std::vector<int>& r_list,
    int block_size,
    float eps,
    std::vector<float*>& d_U_compressed,
    std::vector<float*>& d_Vt_compressed,
    std::vector<size_t>& comp_len_U,
    std::vector<size_t>& comp_len_Vt
);
}  // namespace utils
