#pragma once
#include <cuda_runtime.h>
#include <zfp.h>
#include <vector>

namespace utils {
/**
 * @brief GPU端ZFP压缩（FP32）
 * @param d_input 输入数据（GPU端，float）
 * @param input_len 输入数据长度（元素数）
 * @param eps 压缩精度（控制误差）
 * @param d_output [输出] 压缩后数据（GPU端，float）
 * @param output_len [输出] 压缩后数据长度（字节）
 * @return cudaError_t 错误码
 */
cudaError_t zfp_gpu_compress(
    const float* d_input,
    size_t input_len,
    float eps,
    float*& d_output,
    size_t& output_len
);

/**
 * @brief GPU端ZFP解压缩（FP32）
 * @param d_input 压缩数据（GPU端，float）
 * @param input_len 压缩数据长度（字节）
 * @param eps 压缩精度（需与压缩时一致）
 * @param output_len 解压缩后数据长度（元素数）
 * @param d_output [输出] 解压缩后数据（GPU端，float）
 * @return cudaError_t 错误码
 */
cudaError_t zfp_gpu_decompress(
    const float* d_input,
    size_t input_len,
    float eps,
    size_t output_len,
    float*& d_output
);

/**
 * @brief 批量压缩低秩块基向量（GPU端）
 * @param d_U_list 低秩块U基向量列表（GPU端，float）
 * @param d_Vt_list 低秩块V^T基向量列表（GPU端，float）
 * @param r_list 每个低秩块的秩r
 * @param block_size 超级块尺寸（默认64）
 * @param eps 压缩精度
 * @param d_U_compressed [输出] 压缩后U列表（GPU端）
 * @param d_Vt_compressed [输出] 压缩后V^T列表（GPU端）
 * @param comp_len_U [输出] U压缩后长度（字节）
 * @param comp_len_Vt [输出] V^T压缩后长度（字节）
 * @return cudaError_t 错误码
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