# BLR-Krylov: A Single-GPU Iterative SpMM Framework with Communication Avoidance and Block Low-Rank Optimization

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CUDA Version](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-12-8-download-archive)
[![ZFP Version](https://img.shields.io/badge/ZFP-1.0+-orange.svg)](https://github.com/LLNL/zfp)

## 1. Project Overview
### 1.1 Core Objectives
BLR-Krylov is a high-performance computing framework based on our academic paper. It is specifically designed to address the two core bottlenecks of **iterative Sparse General Matrix Multiplication (SpMM) on single-GPU platforms**:
- **Communication Bottleneck**: Excessive global memory traffic caused by the repeated loading of Krylov subspace basis vectors.
- **Computation Bottleneck**: Redundant arithmetic operations on low-rank blocks (which account for 70%-90% of blocks in scientific computing matrices).

By synergistically integrating **Communication-Avoiding Krylov (CA-Krylov) theory** with the **Block Low-Rank Sparse (BLR-Sparse) format**, this framework achieves dual optimization—"communication reduction + computation compression"—providing efficient computational support for workloads such as Finite Element Method (FEM) analysis, PDE discretization, and Graph Neural Network (GNN) inference.

### 1.2 Core Technical Path
| Technical Module | Core Function | Paper Section |
|------------------|---------------|---------------|
| **BLR-Sparse Preprocessing** | Matrix partitioning (64×64 superblocks), low-rank detection (cross-approximation), basis vector compression (ZFP) | 4.2-4.3 |
| **CA-Krylov Basis Vectors** | Matrix power kernel for batched basis computation (1 load instead of $k$), iterative reuse of valid vectors | 4.4-4.5 |
| **Fused SpMM Execution** | Basis vector inner products for low-rank blocks (80% computation reduction), Tensor Core acceleration for dense blocks (FP16 `mma.sync`) | 4.6-4.8 |
| **Dynamic Iteration Control** | Residual monitoring (≤1e-6), adaptive low-rank threshold tuning ($r_{min}=16 \sim r_{max}=64$) | 4.10 |

## 2. Core Features
1. **Extreme Performance**: Achieves **4.0x–6.0x speedup** over mainstream solutions (KAMI, cuBLAS) on FEM/PDE/GNN workloads. Throughput reaches 520 GFLOPS on 8192×8192 matrix iteration tasks.
2. **Strict Accuracy Guarantee**: Low-rank approximation errors are controlled via ZFP compression (tolerance 1e-6) and residual feedback. The iteration residual remains stably ≤1e-6, meeting the rigorous numerical requirements of scientific computing.
3. **Cross-GPU Portability**: Supports NVIDIA (CUDA), AMD (HIP), and Intel (SYCL) architectures with a minimal performance degradation of only **10%-35%** (far below the 30%-50% degradation of baselines).
4. **High Hardware Utilization**: Achieves **82%-87%** global memory bandwidth utilization and **85%-90%** Tensor Core utilization, fully unlocking GPU hardware potential.
5. **Ease of Use**: Provides complete data preprocessing scripts, example datasets, and visualized performance logs, supporting one-click build and execution.

## 3. Environment Dependencies
### 3.1 Hardware Requirements
| Component | Specification Requirements |
|-----------|----------------------------|
| **GPU** | NVIDIA GPUs prioritized (Hopper/Ada architecture, e.g., GH200, RTX 5090);<br>AMD GPUs require ROCm 6.10+; Intel GPUs require oneAPI 2025+ |
| **CPU** | Multi-core processor (≥8 cores, e.g., Intel Xeon Gold 5220, AMD EPYC 7763) |
| **Memory**| ≥64GB (Required for processing 4096×4096 matrix + 4 basis vectors) |
| **Storage**| ≥10GB (For input matrices, build artifacts, and example datasets) |

### 3.2 Software Dependencies
| Dependency | Version | Purpose | Installation Guide (Ubuntu 22.04) |
|------------|---------|---------|-----------------------------------|
| **CUDA Toolkit** | 12.8 | NVIDIA GPU API, provides cuBLAS/cuSPARSE | [Official Download](https://developer.nvidia.com/cuda-12-8-download-archive) |
| **ZFP** | 1.0+ (w/ GPU) | Compression/decompression of low-rank basis vectors | `git clone https://github.com/LLNL/zfp.git && cd zfp/build && cmake .. -DZFP_WITH_CUDA=ON && make install` |
| **CMake** | ≥3.25 | Build configuration tool | `sudo apt install cmake` |
| **GCC/G++** | ≥9.4 | C++/CUDA compilation (C++17 support) | `sudo apt install gcc-9 g++-9` |
| **Python** | ≥3.8 (Optional) | Matrix format conversion (MAT→CSR), performance data analysis | `pip3 install numpy scipy scikit-sparse` |

## 4. Quick Start
### 4.1 Clone the Project
```bash
git clone [https://github.com/19547035579zz-tech/BLR-Krylov.git](https://github.com/19547035579zz-tech/BLR-Krylov.git)
cd BLR-Krylov
```

### 4.2 Install Dependencies
#### 4.2.1 NVIDIA CUDA (Ubuntu Example)
1. Download and install CUDA 12.8:
   ```bash
   wget [https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_535.104.05_linux.run](https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_535.104.05_linux.run)
   sudo sh cuda_12.8.0_535.104.05_linux.run --toolkit --silent
   ```
2. Configure environment variables (add to `~/.bashrc`):
   ```bash
   echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

#### 4.2.2 ZFP (Enable GPU Support)
```bash
git clone [https://github.com/LLNL/zfp.git](https://github.com/LLNL/zfp.git)
cd zfp && mkdir build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DZFP_WITH_CUDA=ON \
  -DCUDA_ARCHITECTURES=80;90  # Target GH200(90)/RTX5090(89)
make -j$(nproc)
sudo make install
```

### 4.3 Build the Project
```bash
# Create build directory
mkdir build && cd build

# Generate Makefile (Specify CUDA arch: sm_90 for GH200, sm_89 for RTX 5090)
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ARCHITECTURES=90

# Parallel build (threads = CPU cores)
make -j$(nproc)

# Verify compilation (Check for bin/blr_krylov executable)
ls -lh bin/blr_krylov
```

## 5. User Guide
### 5.1 Data Preparation
#### 5.1.1 Input Matrix Format
The framework requires the **CSR (Compressed Sparse Row) binary format**, consisting of 3 files:
| Filename | Format | Description |
|----------|--------|-------------|
| `A_val.bin` | float32 | Non-zero values of the sparse matrix (row-major order) |
| `A_row_ptr.bin` | int32 | Row pointers (`row_ptr[i]` is the index of the first non-zero element of row $i$ in `A_val.bin`) |
| `A_col_idx.bin` | int32 | Column indices corresponding to each non-zero element in `A_val.bin` |

#### 5.1.2 Example Dataset
An FEM 4096×4096 matrix (90% low-rank blocks) is provided for immediate testing:
```bash
# Extract example data (located in data/fem/)
unzip data/fem/fem_4096x4096_csr.zip -d data/fem/
```

#### 5.1.3 Custom Data Conversion
If you use matrices generated by MATLAB (variable name must be `A`), convert them to CSR format using our script:
```bash
python3 data/convert_csr.py \
  --input your_matrix.mat \
  --output_dir data/your_data/
```

### 5.2 Run Commands
#### 5.2.1 Basic Example (FEM 4096×4096 Task)
```bash
# Navigate to the build binary directory
cd build/bin

# Run iterative SpMM (batch size k=4, max iterations=100)
./blr_krylov \
  --matrix-dir ../../data/fem/ \
  --k 4 \
  --iter-max 100 \
  --res-threshold 1e-6
```

#### 5.2.2 Key Parameters Description
| Parameter | Abbr. | Meaning | Default |
|-----------|-------|---------|---------|
| `--matrix-dir` | `-d` | Directory of CSR matrix files (must contain A_val.bin/A_row_ptr.bin/A_col_idx.bin) | `../data/fem` |
| `--k` | `-k` | Number of basis vectors for CA-Krylov batch computation | 4 |
| `--iter-max` | `-i` | Maximum solver iteration steps | 100 |
| `--res-threshold` | `-r` | Residual threshold for convergence (Iteration stops when ≤ this value) | 1e-6 |
| `--r-init` | `-R` | Initial low-rank threshold (Blocks with $r \le r_{init}$ are classified as low-rank) | 32 |
| `--zfp-eps` | `-z` | ZFP compression tolerance (Controls low-rank block error) | 1e-6 |

### 5.3 Result Interpretation
Two main types of information will be output after execution:
#### 5.3.1 Iteration Log
```text
[Iter 0] Residual = 8.2e-3, r_thresh = 32, Valid Basis = 4/4, Time = 40.2ms
[Iter 1] Residual = 1.5e-3, r_thresh = 32, Valid Basis = 4/4, Time = 38.7ms
...
[Iter 23] Residual = 9.8e-7, r_thresh = 28, Valid Basis = 3/4, Time = 37.5ms
Iteration finished! Final residual = 9.8e-7, Total time = 920ms
```

#### 5.3.2 Performance Report
```text
Performance Summary:
- Throughput: 480 GFLOPS
- Global Memory Bandwidth Utilization: 85.2%
- Tensor Core Utilization: 88.7%
- Speedup vs KAMI: 5.8x
- Space Overhead Reduction: 72% (compared to Z-Morton format)
```

## 6. Project Structure
```text
BLR-Krylov/
├── include/                  # Headers: API and Data Structures
│   ├── blr_krylov_types.h    # Core structures (BLRMatrix, BasisVectors, etc.)
│   ├── blr_sparse_preprocess.h # BLR Preprocessing APIs
│   ├── ca_krylov_basis.h     # CA-Krylov Basis Vector APIs
│   ├── fused_spgemm.h        # Fused SpMM APIs
│   ├── iter_controller.h     # Iteration Control APIs
│   └── utils/                # Utilities (ZFP, Morton Encoding, etc.)
├── src/                      # Source: Core Logic Implementation
│   ├── blr_sparse_preprocess.cu # BLR Preprocessing (Partitioning, Low-rank detection)
│   ├── ca_krylov_basis.cu    # CA-Krylov Basis (Matrix power kernel, Reuse)
│   ├── fused_spgemm.cu       # Fused SpMM (Low-rank inner products, Tensor Cores)
│   ├── iter_controller.cu    # Iteration Control (Residuals, Threshold tuning)
│   ├── utils/                # Utilities implementation
│   └── main.cu               # Main: Coordinates modules for execution
├── data/                     # Data Directory
│   ├── fem/                  # FEM example dataset (4096×4096)
│   └── convert_csr.py        # MAT to CSR conversion script
├── cmake/                    # CMake Helper Scripts
│   └── FindZFP.cmake         # Find ZFP library (with GPU support)
└── CMakeLists.txt            # Project build configuration
```

## 7. FAQ
### Q1: Compilation error "ZFP GPU support not found"?
A1: Recompile ZFP making sure to enable `-DZFP_WITH_CUDA=ON`, and verify that `/usr/local/include/zfp_cuda.h` exists.

### Q2: Low GPU utilization (<50%) during runtime?
A2: Check two things:
1. Is the proportion of low-rank blocks in the input matrix ≥ 70%?
2. Is the basis vector count `k` too small? (Recommended: 4~8).

### Q3: Build fails across AMD/Intel GPUs?
A3: 
- **AMD**: Replace `CUDA` dependencies in `CMakeLists.txt` with `HIP`, and change the compiler to `hipcc`.
- **Intel**: Use the `DPC++` compiler (`dpcpp`), depend on `oneAPI`, and adjust the superblock size to 32×32 (to fit the 64KB shared memory limit).

## 8. Citation & Acknowledgements
### 8.1 Citation
If you use this project in your research, please cite our paper accepted by ACM TACO:
```bibtex
@article{li2026blr,
  title={BLR-Krylov: A Single-GPU Iterative SpMM Framework with Communication Avoidance and Block Low-Rank Optimization},
  author={Li, WenTao and Zhang, Zheng and Zhao, Jie and Zhu, SongQuan and Hui, Ming},
  journal={ACM Transactions on Architecture and Code Optimization (TACO)},
  year={2026},
  publisher={ACM}
}
```

### 8.2 Acknowledgements
- [ZFP](https://github.com/LLNL/zfp): Library for compressing low-rank block basis vectors.
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit): NVIDIA GPU programming ecosystem.
- [Intel oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi.html): Cross-architecture compute support.

## 9. Contact
- For bug reports or feature requests, please open an Issue. Code updates will be regularly synchronized.

---
---
## Appendix: Script Documentation

### I. FEM 4096×4096 Example Dataset Generation Script (`generate_fem_data.py`)
**Dataset Generation Steps:**
1. Install dependencies: `pip3 install numpy scipy`
2. Run the script: `python3 data/fem/generate_fem_data.py`
*(Generates approx. 80MB of CSR binary files matching the exact features used in our paper)*

### II. MAT to CSR Binary Conversion Script (`convert_csr.py`)
This script converts MATLAB sparse matrices (`.mat` files, variable name `A`) into the CSR binary format.
```bash
# Convert a custom PDE matrix
python3 data/convert_csr.py \
  --input ./pde_matrix.mat \
  --output-dir ./data/pde/ \
  --var-name A
```
