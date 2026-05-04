#include "utils/zfp_gpu.cuh"
#include <cuda_runtime_api.h>
#include <cassert>

namespace utils {
// GPU端ZFP压缩核（FP32，简化版，实际需调用ZFP GPU接口）
__global__ void zfp_compress_kernel(
    const float* d_input,
    size_t input_len,
    float eps,
    float* d_output,
    size_t* d_output_len
) {
    // 实际实现需调用ZFP的cudaCompress函数，此处为框架示例
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        // 简化：按精度估算压缩长度（实际需调用zfp_stream_maximum_size）
        *d_output_len = static_cast<size_t>(input_len * sizeof(float) * 0.3);  // 3:1压缩比
        for (size_t i = 0; i < *d_output_len / sizeof(float); i++) {
            d_output[i] = d_input[i] * (1.0f + eps);  // 模拟压缩误差
        }
    }
}

// GPU端ZFP解压缩核（FP32）
__global__ void zfp_decompress_kernel(
    const float* d_input,
    size_t input_len,
    float eps,
    size_t output_len,
    float* d_output
) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < output_len) {
        // 简化：恢复压缩前数据（实际需调用zfp_stream_decompress）
        d_output[tid] = d_input[tid] / (1.0f + eps);
    }
}

cudaError_t zfp_gpu_compress(
    const float* d_input,
    size_t input_len,
    float eps,
    float*& d_output,
    size_t& output_len
) {
    if (!d_input || input_len == 0) return cudaErrorInvalidDevicePointer;

    // 步骤1：估算压缩后长度（实际需调用ZFP API）
    size_t max_comp_len = static_cast<size_t>(input_len * sizeof(float) * 0.5);  // 预留空间
    cudaMalloc(&d_output, max_comp_len);
    if (!d_output) return cudaErrorMemoryAllocation;

    // 步骤2：GPU压缩（简化版，实际需链接ZFP GPU库）
    size_t* d_comp_len;
    cudaMalloc(&d_comp_len, sizeof(size_t));
    dim3 grid(1), block(1);
    zfp_compress_kernel<<<grid, block>>>(d_input, input_len, eps, d_output, d_comp_len);
    cudaMemcpy(&output_len, d_comp_len, sizeof(size_t), cudaMemcpyDeviceToHost);

    // 步骤3：释放临时内存
    cudaFree(d_comp_len);
    return cudaGetLastError();
}

cudaError_t zfp_gpu_decompress(
    const float* d_input,
    size_t input_len,
    float eps,
    size_t output_len,
    float*& d_output
) {
    if (!d_input || input_len == 0 || output_len == 0) return cudaErrorInvalidDevicePointer;

    // 步骤1：分配解压缩内存
    cudaMalloc(&d_output, output_len * sizeof(float));
    if (!d_output) return cudaErrorMemoryAllocation;

    // 步骤2：GPU解压缩
    dim3 grid((output_len + 255) / 256), block(256);
    zfp_decompress_kernel<<<grid, block>>>(d_input, input_len, eps, output_len, d_output);
    return cudaGetLastError();
}

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
) {
    const size_t block_cnt = d_U_list.size();
    assert(block_cnt == d_Vt_list.size() && block_cnt == r_list.size());

    d_U_compressed.resize(block_cnt, nullptr);
    d_Vt_compressed.resize(block_cnt, nullptr);
    comp_len_U.resize(block_cnt, 0);
    comp_len_Vt.resize(block_cnt, 0);

    // 批量压缩每个低秩块的U和V^T
    for (size_t b = 0; b < block_cnt; b++) {
        const int r = r_list[b];
        const size_t u_len = block_size * r;    // U: B×r
        const size_t vt_len = r * block_size;  // V^T: r×B

        // 压缩U
        CUDA_CHECK(zfp_gpu_compress(d_U_list[b], u_len, eps, d_U_compressed[b], comp_len_U[b]));
        // 压缩V^T
        CUDA_CHECK(zfp_gpu_compress(d_Vt_list[b], vt_len, eps, d_Vt_compressed[b], comp_len_Vt[b]));
    }
    return cudaSuccess;
}
}  // namespace utils