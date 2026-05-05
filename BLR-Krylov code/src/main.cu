#include "blr_krylov_types.h"
#include "blr_sparse_preprocess.h"
#include "ca_krylov_basis.h"
#include "fused_spgemm.h"
#include "iter_controller.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <iostream>
#include <string>
#include <getopt.h>

#define CUDA_CHECK(err) do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

struct CmdArgs {
    std::string matrix_dir = "../data/fem/";
    size_t k = 4;
    int iter_max = 100;
    float res_threshold = 1e-6;
    int r_init = 32;
    int super_block_size = 64;
    int sub_block_size = 16;
    float zfp_eps = 1e-6;
};

void parse_cmd_args(int argc, char** argv, CmdArgs& args) {
    const struct option long_opts[] = {
        {"matrix-dir", required_argument, nullptr, 'd'},
        {"k", required_argument, nullptr, 'k'},
        {"iter-max", required_argument, nullptr, 'i'},
        {"res-threshold", required_argument, nullptr, 'r'},
        {"r-init", required_argument, nullptr, 'R'},
        {"super-block", required_argument, nullptr, 'S'},
        {"sub-block", required_argument, nullptr, 's'},
        {"zfp-eps", required_argument, nullptr, 'z'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "d:k:i:r:R:S:s:z:", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'd': args.matrix_dir = optarg; break;
            case 'k': args.k = static_cast<size_t>(atoi(optarg)); break;
            case 'i': args.iter_max = atoi(optarg); break;
            case 'r': args.res_threshold = atof(optarg); break;
            case 'R': args.r_init = atoi(optarg); break;
            case 'S': args.super_block_size = atoi(optarg); break;
            case 's': args.sub_block_size = atoi(optarg); break;
            case 'z': args.zfp_eps = atof(optarg); break;
            default:
                fprintf(stderr, "Usage: %s [--matrix-dir DIR] [--k K] [--iter-max MAX] [--res-threshold THR]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }
}

void generate_random_vec(size_t M, float*& d_vec) {
    std::vector<float> h_vec(M);
    for (size_t i = 0; i < M; i++) {
        h_vec[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    cudaMalloc(&d_vec, M * sizeof(float));
    cudaMemcpy(d_vec, h_vec.data(), M * sizeof(float), cudaMemcpyHostToDevice);
}

int main(int argc, char** argv) {
    CmdArgs args;
    parse_cmd_args(argc, argv, args);
    srand(static_cast<unsigned int>(time(nullptr)));

    int dev_id = 0;
    CUDA_CHECK(cudaSetDevice(dev_id));
    cublasHandle_t cublas_h;
    cusparseHandle_t cusparse_h;
    CUDA_CHECK(cublasCreate(&cublas_h));
    CUDA_CHECK(cusparseCreate(&cusparse_h));

    HostCSRMatrix host_csr;
    const std::string val_path = args.matrix_dir + "/A_val.bin";
    const std::string row_ptr_path = args.matrix_dir + "/A_row_ptr.bin";
    const std::string col_idx_path = args.matrix_dir + "/A_col_idx.bin";
    if (!blr_load_csr_from_bin(val_path, row_ptr_path, col_idx_path, host_csr)) {
        fprintf(stderr, "Failed to load CSR matrix from %s\n", args.matrix_dir.c_str());
        exit(EXIT_FAILURE);
    }
    printf("Loaded CSR matrix: %zu×%zu, NNZ = %zu\n", host_csr.rows, host_csr.cols, host_csr.nnz);

    BLRMatrix blr_A;
    CUDA_CHECK(blr_sparse_preprocess(
        cusparse_h, host_csr, args.r_init,
        args.super_block_size, args.sub_block_size, args.zfp_eps, blr_A
    ));
    printf("BLR-Sparse preprocess done: %zu blocks (lowrank: %zu, dense: %zu, zero: %zu)\n",
           blr_A.total_blocks,
           blr_A.d_U_compressed.size(),
           blr_A.dense_morton.size(),
           blr_A.total_blocks - blr_A.d_U_compressed.size() - blr_A.dense_morton.size());

    std::vector<float*> d_U_decomp, d_Vt_decomp;
    CUDA_CHECK(blr_decompress_lowrank_blocks(blr_A, args.zfp_eps, d_U_decomp, d_Vt_decomp));

    float *d_V0, *d_b, *d_x;
    generate_random_vec(host_csr.rows, d_V0);
    generate_random_vec(host_csr.rows, d_b);
    cudaMalloc(&d_x, host_csr.rows * sizeof(float));

    IterParams iter_params = iter_init_params(
        args.res_threshold, args.r_init, 64, 16, args.iter_max, args.zfp_eps
    );

    printf("Starting iteration (k = %zu, res_thresh = %.2e)\n", args.k, args.res_threshold);
    CUDA_CHECK(iter_main_loop(
        cublas_h, cusparse_h, blr_A, d_U_decomp, d_Vt_decomp,
        d_V0, d_b, d_x, iter_params, args.k
    ));

    float final_res;
    CUDA_CHECK(iter_compute_residual(
        cublas_h, blr_A, d_U_decomp, d_Vt_decomp, d_x, d_b, final_res
    ));
    printf("Iteration finished! Final residual = %.2e\n", final_res);

    blr_A.free();
    blr_free_decompressed_blocks(d_U_decomp, d_Vt_decomp);
    cudaFree(d_V0);
    cudaFree(d_b);
    cudaFree(d_x);
    cublasDestroy(cublas_h);
    cusparseDestroy(cusparse_h);
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
