BLR-Krylov: A Single-GPU Iterative SpMM Framework with Communication Avoidance and Block Low-Rank Optimization1. Project Overview1.1 Core ObjectivesBLR-Krylov is a high-performance computing framework based on an academic paper, specifically designed to resolve two core bottlenecks in iterative Sparse Matrix-Matrix Multiplication (SpMM) on a single GPU platform:Communication Bottleneck: Global memory access overhead caused by the frequent loading of Krylov subspace basis vectors.Computation Bottleneck: Redundant arithmetic operations in low-rank blocks (which account for 70%-90% of scientific computing matrices).By integrating Communication-Avoiding Krylov (CA-Krylov) theory with the Block Low-Rank Sparse (BLR-Sparse) format, this framework achieves a dual optimization of "communication reduction + computation compression." It provides highly efficient computational support for scenarios such as Finite Element Analysis (FEM), Partial Differential Equation (PDE) discretization, and Graph Neural Network (GNN) inference.1.2 Core Technical PathTechnical ModuleCore FunctionPaper SectionBLR-Sparse PreprocessingMatrix partitioning (64x64 superblocks), low-rank detection (cross approximation), basis vector compression (ZFP).4.2-4.3CA-Krylov Basis VectorsBatched computation of basis vectors using a matrix power kernel (1 load replaces k loads), iterative reuse of valid vectors.4.4-4.5Fused SpMM ExecutionBasis vector inner products for low-rank blocks (80% computation reduction), Tensor Core acceleration for dense blocks (FP16 mma.sync).4.6-4.8Dynamic Iteration ControlResidual monitoring (≤1e-6), adaptive adjustment of the low-rank threshold (r_min=16 to r_max=64).4.102. Core FeaturesExtreme Performance: Achieves 4.0x-6.0x speedups over mainstream solutions (KAMI, cuBLAS) under FEM/PDE/GNN workloads. Throughput reaches 520 GFLOPS for 8192x8192 matrix iteration tasks.Strict Precision Guarantee: Low-rank approximation errors are strictly managed via ZFP compression (precision 1e-6) and residual feedback control, keeping the iterative residual stably ≤1e-6, meeting the numerical requirements of scientific computing.Cross-GPU Portability: Supports NVIDIA (CUDA), AMD (HIP), and Intel (SYCL) architectures, with a performance degradation of only 10%-35% (far lower than the 30%-50% baseline).High Hardware Utilization: Global memory bandwidth utilization reaches 82%-87%, and Tensor Core utilization reaches 85%-90%, fully exploiting GPU hardware potential.Ease of Use: Provides comprehensive data preprocessing scripts, sample datasets, and visualized performance logs, supporting one-click build and execution.3. Environment Dependencies3.1 Hardware RequirementsHardwareSpecificationsGPUPriority support for NVIDIA GPUs (Hopper/Ada architectures, e.g., GH200, RTX 5090). AMD GPUs require ROCm 6.10+; Intel GPUs require oneAPI 2025+.CPUMulti-core processor (≥8 cores, e.g., Intel Xeon Gold 5220, AMD EPYC 7763).Memory≥64GB (Required for handling 4096×4096 matrices + 4 basis vectors).Storage≥10GB (For storing input matrices, build artifacts, and sample data).3.2 Software DependenciesDependencyVersionPurposeInstallation (Ubuntu 22.04)CUDA Toolkit12.8NVIDIA GPU programming interface, provides cuBLAS/cuSPARSEOfficial DownloadZFP1.0+ (with GPU)Low-rank block basis vector compression/decompressiongit clone https://github.com/LLNL/zfp.git && cd zfp/build && cmake .. -DZFP_WITH_CUDA=ON && make installCMake≥3.25Project build configuration toolsudo apt install cmakeGCC/G++≥9.4C++/CUDA compilation (C++17 support required)sudo apt install gcc-9 g++-9Python≥3.8 (Optional)Matrix format conversion (MAT to CSR), performance data analysispip3 install numpy scipy scikit-sparse4. Quick Start4.1 Clone the Projectgit clone [https://github.com/yourusername/BLR-Krylov.git](https://github.com/yourusername/BLR-Krylov.git)
cd BLR-Krylov
4.2 Install Dependencies4.2.1 NVIDIA CUDA (Ubuntu Example)Download and install CUDA 12.8:wget [https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_535.104.05_linux.run](https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_535.104.05_linux.run)
sudo sh cuda_12.8.0_535.104.05_linux.run --toolkit --silent
Configure environment variables (add to ~/.bashrc):echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
4.2.2 ZFP (Enable GPU Support)git clone [https://github.com/LLNL/zfp.git](https://github.com/LLNL/zfp.git)
cd zfp && mkdir build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DZFP_WITH_CUDA=ON \
  -DCUDA_ARCHITECTURES=80;90  # Target architectures, e.g., GH200(90), RTX 5090(89)
make -j$(nproc)
sudo make install
4.3 Build the Project# Create build directory
mkdir build && cd build

# Generate Makefile (Specify CUDA architecture, e.g., sm_90 for GH200, sm_89 for RTX 5090)
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ARCHITECTURES=90

# Compile in parallel (threads = CPU cores)
make -j$(nproc)

# Verify build output (generates bin/blr_krylov executable)
ls -lh bin/blr_krylov
5. Usage Guide5.1 Data Preparation5.1.1 Input Matrix FormatThe framework requires the CSR (Compressed Sparse Row) binary format, which consists of 3 files:FilenameFormatDescriptionA_val.binfloat32Non-zero element values of the sparse matrix (stored in row-major order).A_row_ptr.binint32Row pointers (row_ptr[i] is the index in A_val.bin for the first non-zero element in row i).A_col_idx.binint32Column indices of the non-zero elements (corresponding exactly to A_val.bin).5.1.2 Sample DatasetWe provide an FEM 4096x4096 matrix (90% low-rank blocks) for out-of-the-box testing:# Extract the sample data (located in data/fem/)
unzip data/fem/fem_4096x4096_csr.zip -d data/fem/
5.1.3 Custom Data ConversionIf using MATLAB-generated matrices (variable name must be A), convert them to CSR format via our script:python3 data/convert_csr.py \
  --input your_matrix.mat \
  --output_dir data/your_data/
5.2 Run Commands5.2.1 Basic Example (FEM 4096x4096 workload)cd build/bin

./blr_krylov \
  --matrix-dir ../../data/fem/ \
  --k 4 \
  --iter-max 100 \
  --res-threshold 1e-6
5.2.2 Key Parameter DescriptionsParameterShortDescriptionDefault--matrix-dir-dDirectory of CSR matrix files../data/fem--k-kNumber of basis vectors for CA-Krylov batched computation4--iter-max-iMaximum number of iteration steps100--res-threshold-rResidual threshold for convergence1e-6--r-init-RInitial low-rank threshold32--zfp-eps-zZFP compression precision1e-65.3 Result Interpretation5.3.1 Iteration Log[Iter 0] Residual = 8.2e-3, r_thresh = 32, Valid Basis = 4/4, Time = 40.2ms
[Iter 1] Residual = 1.5e-3, r_thresh = 32, Valid Basis = 4/4, Time = 38.7ms
...
[Iter 23] Residual = 9.8e-7, r_thresh = 28, Valid Basis = 3/4, Time = 37.5ms
Iteration finished! Final residual = 9.8e-7, Total time = 920ms
5.3.2 Performance ReportPerformance Summary:
- Throughput: 480 GFLOPS
- Global Memory Bandwidth Utilization: 85.2%
- Tensor Core Utilization: 88.7%
- Speedup vs KAMI: 5.8x
- Space Overhead Reduction: 72% (vs. Z-Morton format)
6. Project StructureBLR-Krylov/
├── include/                  # Headers
│   ├── blr_krylov_types.h
│   ├── blr_sparse_preprocess.h
│   ├── ca_krylov_basis.h
│   ├── fused_spgemm.h
│   ├── iter_controller.h
│   └── utils/
├── src/                      # Source files
│   ├── blr_sparse_preprocess.cu
│   ├── ca_krylov_basis.cu
│   ├── fused_spgemm.cu
│   ├── iter_controller.cu
│   ├── utils/
│   └── main.cu
├── data/                     # Data directory
│   ├── fem/
│   └── convert_csr.py
├── cmake/                    # CMake helper scripts
│   └── FindZFP.cmake
└── CMakeLists.txt            # Build configuration
7. FAQ (Frequently Asked Questions)Q1: Compilation error "ZFP GPU support not found"?A1: When rebuilding ZFP, ensure you enable -DZFP_WITH_CUDA=ON and verify that the file /usr/local/include/zfp_cuda.h exists.Q2: Low GPU utilization (<50%) during execution?A2: Check two aspects:Ensure the low-rank block ratio of the input matrix is ≥70%.Verify if the number of basis vectors k is too small. We recommend setting it to 4~8.Q3: Cross-compilation for AMD/Intel GPUs fails?A3: - AMD: Replace CUDA dependencies with HIP in CMakeLists.txt and change the compiler to hipcc.Intel: Use the DPC++ compiler (dpcpp) with oneAPI dependencies. Adjust the superblock size to 32x32.Q4: Residual exceeds the threshold (>1e-6) and fails to converge?A4: Increase the initial low-rank threshold --r-init (e.g., from 32 to 40) or lower the ZFP compression precision tolerance --zfp-eps (e.g., from 1e-6 to 1e-7).8. Citation & Acknowledgements8.1 CitationIf you use this project in your research, please cite our paper:@inproceedings{BLR-Krylov-2025,
  title={BLR-Krylov: A Single-GPU Iterative SpMM Framework with Communication Avoidance and Block Low-Rank Optimization},
  author={Li, Wentao and Zhang, Zheng and Zhao, Jie and Zhu, Songquan and Hui, Ming},
  booktitle={Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC)},
  year={2025}
}
8.2 AcknowledgementsZFP: Library used for low-rank block basis vector compression.CUDA Toolkit: NVIDIA GPU programming ecosystem.Intel oneAPI: Cross-architecture computation support.9. ContactCode Updates: We regularly sync supplementary experiments and performance optimizations from the paper. We encourage you to Star the project to receive update notifications.Appendix: Data ScriptsI. FEM 4096x4096 Sample Dataset Generation Script (generate_fem_data.py)Since providing a 4096x4096 matrix binary file directly (approx. 100MB+) is inconvenient for transfer, the following script generates an FEM sparse matrix matching the paper's characteristics (15% non-zero density, 90% low-rank block ratio, 64x64 superblock partitioning). Once generated, it automatically saves as CSR binary formats (A_val.bin, A_row_ptr.bin, A_col_idx.bin), fully compatible with the C++ framework.1. Script UsageInstall dependencies:pip3 install numpy scipy
Run the script to generate data:python3 data/fem/generate_fem_data.py
II. MAT to CSR Binary Script (convert_csr.py)This script is used to convert MATLAB-generated sparse matrices (.mat files, variable name must be A) into the CSR binary format supported by the C++ code. It supports custom input/output paths, making it adaptable for custom PDE or GNN datasets.1. Usage Guide1.1 Basic Usage (Converting MAT files)Prepare MAT File: Ensure the sparse matrix variable saved in MATLAB is named A (or specify another name via --var-name).Run the conversion script:python3 data/convert_csr.py \
  --input ./pde_matrix.mat \
  --output-dir ./data/pde/ \
  --var-name A
1.2 Supported Matrix TypesSparse Matrices: Matrices created using sparse() in MATLAB.Dense Matrices: Dense matrices created using zeros() or randn() in MATLAB.Data Types: Supports float32/float64; the script automatically normalizes outputs to float32.III. Data Compatibility NotesCompatibility with the C++ Framework: The generated CSR binary files can be read directly via the blr_load_csr_from_bin() function (src/blr_sparse_preprocess.cu).FEM Dataset Features: The low-rank block ratio and non-zero density of the FEM dataset match exactly with the experimental conditions described in Section 5.2 of the paper.