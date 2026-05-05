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
2. **Strict Accuracy Guarantee**: Low-rank approximation errors are strictly managed via ZFP compression (precision 1e-6) and residual feedback control, keeping the iterative residual stably ≤1e-6, meeting the numerical requirements of scientific computing.
3. **Cross-GPU Portability**: Supports NVIDIA (CUDA), AMD (HIP), and Intel (SYCL) architectures, with a performance degradation of only **10%-35%** (far lower than the 30%-50% baseline).
4. **High Hardware Utilization**: Global memory bandwidth utilization reaches **82%-87%**, and Tensor Core utilization reaches **85%-90%**, fully exploiting GPU hardware potential.
5. **Ease of Use**: Provides comprehensive data preprocessing scripts, sample datasets, and visualized performance logs, supporting one-click build and execution.

## 3. Environment Dependencies
### 3.1 Hardware Requirements
| Component | Specification Requirements |
|-----------|----------------------------|
| **GPU** | Priority support for NVIDIA GPUs (Hopper/Ada architectures, e.g., GH200, RTX 5090);<br>AMD GPUs require ROCm 6.10+; Intel GPUs require oneAPI 2025+ |
| **CPU** | Multi-core processor (≥8 cores, e.g., Intel Xeon Gold 5220, AMD EPYC 7763) |
| **Memory**| ≥64GB (Required for handling 4096×4096 matrices + 4 basis vectors) |
| **Storage**| ≥10GB (For storing input matrices, build artifacts, and sample data) |

### 3.2 Software Dependencies
| Dependency | Version | Purpose | Installation Guide (Ubuntu 22.04) |
|------------|---------|---------|-----------------------------------|
| **CUDA Toolkit** | 12.8 | NVIDIA GPU programming interface, provides cuBLAS/cuSPARSE | [Official Download](https://developer.nvidia.com/cuda-12-8-download-archive) |
| **ZFP** | 1.0+ (with GPU) | Low-rank block basis vector compression/decompression | `git clone https://github.com/LLNL/zfp.git && cd zfp/build && cmake .. -DZFP_WITH_CUDA=ON && make install` |
| **CMake** | ≥3.25 | Project build configuration tool | `sudo apt install cmake` |
| **GCC/G++** | ≥9.4 | C++/CUDA compilation (C++17 support required) | `sudo apt install gcc-9 g++-9` |
| **Python** | ≥3.8 (Optional) | Matrix format conversion (MAT to CSR), performance data analysis | `pip3 install numpy scipy scikit-sparse` |

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
| `A_val.bin` | float32 | Non-zero elements of the sparse matrix (row-major order) |
| `A_row_ptr.bin` | int32 | Row pointers (`row_ptr[i]` is the index of the first non-zero element of row $i$ in `A_val.bin`) |
| `A_col_idx.bin` | int32 | Column indices corresponding to `A_val.bin` |

#### 5.1.2 Example Dataset
An FEM 4096×4096 matrix (90% low-rank blocks) is provided. You can use it directly:
```bash
# Extract example data (located in data/fem/)
unzip data/fem/fem_4096x4096_csr.zip -d data/fem/
```

#### 5.1.3 Custom Data Conversion
If you use matrices generated by MATLAB (variable name must be `A`), convert them to CSR format using the provided script:
```bash
python3 data/convert_csr.py \
  --input your_matrix.mat \       # Input MAT file path
  --output_dir data/your_data/    # Output CSR directory
```

### 5.2 Run Commands
#### 5.2.1 Basic Example (FEM 4096×4096 Task)
```bash
# Navigate to the compiled binary directory
cd build/bin

# Run iterative SpGEMM (basis vector count k=4, max iterations=100)
./blr_krylov \
  --matrix-dir ../../data/fem/ \  # CSR matrix directory
  --k 4 \                         # Number of basis vectors (Recommended 4~8)
  --iter-max 100 \                # Maximum iteration steps
  --res-threshold 1e-6            # Residual convergence threshold
```

#### 5.2.2 Key Parameters Description
| Parameter | Abbr. | Meaning | Default |
|-----------|-------|---------|---------|
| `--matrix-dir` | `-d` | Directory of CSR matrix files (must contain A_val.bin/A_row_ptr.bin/A_col_idx.bin) | `../data/fem` |
| `--k` | `-k` | Number of basis vectors for CA-Krylov batched computation | 4 |
| `--iter-max` | `-i` | Maximum steps for the iterative solver | 100 |
| `--res-threshold` | `-r` | Residual threshold for convergence (Stops when ≤ this value) | 1e-6 |
| `--r-init` | `-R` | Initial low-rank threshold (Blocks with $r \le r_{init}$ are low-rank) | 32 |
| `--zfp-eps` | `-z` | ZFP compression precision (Controls low-rank block error) | 1e-6 |

### 5.3 Result Interpretation
Two main types of key information are output after execution:
#### 5.3.1 Iteration Log
```text
[Iter 0] Residual = 8.2e-3, r_thresh = 32, Valid Basis = 4/4, Time = 40.2ms
[Iter 1] Residual = 1.5e-3, r_thresh = 32, Valid Basis = 4/4, Time = 38.7ms
...
[Iter 23] Residual = 9.8e-7, r_thresh = 28, Valid Basis = 3/4, Time = 37.5ms
Iteration finished! Final residual = 9.8e-7, Total time = 920ms
```
- `Residual`: Current iteration residual (Needs to be ≤ 1e-6).
- `r_thresh`: Dynamically adjusted low-rank threshold.
- `Valid Basis`: Number of valid basis vectors (Reuse mechanism reduces communication).

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
├── include/                  # Headers: Interface and data structure definitions
│   ├── blr_krylov_types.h    # Core structures (BLRMatrix, BasisVectors, etc.)
│   ├── blr_sparse_preprocess.h # BLR preprocessing interface
│   ├── ca_krylov_basis.h     # CA-Krylov basis vector interface
│   ├── fused_spgemm.h        # Fused SpGEMM interface
│   ├── iter_controller.h     # Iteration control interface
│   └── utils/                # Utility interfaces (ZFP, Morton Encoding, etc.)
├── src/                      # Source: Core logic implementation
│   ├── blr_sparse_preprocess.cu # BLR preprocessing (Partitioning, low-rank detection)
│   ├── ca_krylov_basis.cu    # CA-Krylov basis vectors (Matrix power kernel, reuse)
│   ├── fused_spgemm.cu       # Fused SpGEMM (Low-rank inner products, Tensor Core)
│   ├── iter_controller.cu    # Iteration control (Residual, threshold adjustment)
│   ├── utils/                # Utility implementations
│   └── main.cu               # Main program: Coordinates module execution
├── data/                     # Data directory
│   ├── fem/                  # FEM example dataset (4096×4096)
│   └── convert_csr.py        # MAT to CSR format conversion script
├── cmake/                    # CMake helper scripts
│   └── FindZFP.cmake         # Find ZFP library (with GPU support)
└── CMakeLists.txt            # Project build configuration
```

## 7. FAQ
### Q1: "ZFP GPU support not found" error during compilation?
A1: When recompiling ZFP, ensure `-DZFP_WITH_CUDA=ON` is enabled, and verify that `/usr/local/include/zfp_cuda.h` exists.

### Q2: Low GPU utilization (<50%) during runtime?
A2: Check two points:
1. Is the proportion of low-rank blocks in the input matrix ≥ 70%? (Analyze using `data/analyze_lowrank.py`).
2. Is the number of basis vectors `k` too small? (Recommended to set to 4~8 to balance arithmetic intensity and memory overhead).

### Q3: Compilation fails across AMD/Intel GPUs?
A3: 
- **AMD**: Replace `CUDA` related dependencies in `CMakeLists.txt` with `HIP`, and change the compiler to `hipcc`.
- **Intel**: Use the `DPC++` compiler (`dpcpp`), depend on `oneAPI`, and adjust the superblock size to 32×32 (to fit 64KB shared memory).

### Q4: Residual exceeds the limit (>1e-6)?
A4: Increase the initial low-rank threshold `--r-init` (e.g., from 32 to 40), or decrease the ZFP compression precision `--zfp-eps` (e.g., from 1e-6 to 1e-7).

## 8. Citation & Acknowledgements
### 8.1 Citation
If you use this project in your research, please cite our paper:
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
- [ZFP](https://github.com/LLNL/zfp): Low-rank block basis vector compression library.
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit): NVIDIA GPU programming ecosystem.
- [Intel oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi.html): Cross-architecture computing support.

## 9. Contact
- Code Updates: We regularly synchronize supplementary experiments and performance optimizations from the paper. We recommend Starring the project to receive update notifications.

---
---

## Appendix: Script Documentation

### I. FEM 4096×4096 Example Dataset Generation Script (`generate_fem_data.py`)
Since directly providing a 4096×4096 matrix binary file (approx. 100MB+) is inconvenient for transfer, the following script generates an FEM sparse matrix conforming to the paper's characteristics (15% density, 90% low-rank block ratio, 64×64 superblock partitioning). After generation, it is automatically saved in CSR binary format (`A_val.bin`/`A_row_ptr.bin`/`A_col_idx.bin`), seamlessly compatible with the aforementioned code.

**Dataset Generation Steps:**
1. Install dependencies: `pip3 install numpy scipy`
2. Run script to generate dataset: `python3 data/fem/generate_fem_data.py`

**Generation Results:**
- Output directory: `data/fem/`
- File size: Approx. 80MB (`A_val.bin`≈60MB, `A_row_ptr.bin`≈16KB, `A_col_idx.bin`≈20MB)
- Matrix features: 4096×4096, 15% non-zero rate, 90% low-rank block (64×64) ratio, perfectly matching the paper's experimental conditions.

### II. MAT File to CSR Binary Conversion Script (`convert_csr.py`)
This script is used to convert sparse matrices generated by MATLAB (`.mat` files, variable name must be `A`) to the CSR binary format supported by the aforementioned code. It supports custom input/output paths, adapting to custom datasets like PDE, GNN, etc.

**2.1 Basic Usage (Convert MAT file)**
- Prepare MAT file: Ensure the sparse matrix variable saved in MATLAB is named `A` (or specify another name via `--var-name`).
- Run conversion script:
```bash
# Convert custom PDE matrix (input is MAT file path, output is output directory)
python3 data/convert_csr.py \
  --input ./pde_matrix.mat \
  --output-dir ./data/pde/ \
  --var-name A  # Modify this parameter if the matrix variable name is not A
```

**2.2 Supported Matrix Types**
- Sparse matrices: Matrices created with `sparse()` in MATLAB (e.g., `A = sparse(i,j,v,m,n)`).
- Dense matrices: Dense matrices created with `zeros()`/`randn()` in MATLAB (the script will automatically convert to sparse format, filtering out zero elements).
- Data types: Supports `float32`/`float64`, the script will automatically unify conversion to `float32`.

**2.3 Conversion Verification**
After conversion is complete, file integrity can be verified via the following command:
```bash
# View output directory files
ls -lh ./data/pde/
# Expected output: A_val.bin, A_row_ptr.bin, A_col_idx.bin
```

### III. Data Compatibility Notes
**Compatibility with the aforementioned code:**
- The generated CSR binary files can be read directly via the `blr_load_csr_from_bin()` function (`src/blr_sparse_preprocess.cu`).
- The FEM dataset's low-rank block ratio, non-zero rate, and other features perfectly match the experimental conditions in Section 5.2 of the paper.

**Custom Dataset Adaptation:**
- If the generated matrix dimension is not 4096×4096, ensure the `--super-block` parameter (superblock size) can evenly divide the matrix dimension when running `blr_krylov` (e.g., for a 2048×2048 matrix, recommend setting `--super-block 64`).
- When the low-rank block ratio is below 70%, performance can be improved by increasing `--k` (number of basis vectors) (recommend setting to 6~8).
