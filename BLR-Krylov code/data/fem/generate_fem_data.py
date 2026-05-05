import numpy as np
import scipy.sparse as sp
import struct
import os

def generate_fem_stiffness_matrix(
    size: int = 4096,
    non_zero_ratio: float = 0.15,
    low_rank_block_ratio: float = 0.9,
    super_block_size: int = 64,
    rank: int = 24
) -> sp.csr_matrix:
    data = []
    row_ind = []
    col_ind = []

    block_cnt = size // super_block_size
    assert block_cnt * super_block_size == size, "Size must be a multiple of super_block_size"

    for p in range(block_cnt):
        for q in range(block_cnt):
            row_start = p * super_block_size
            col_start = q * super_block_size

            if p == q:
                dense_block = np.random.randn(super_block_size, super_block_size)
                mask = np.random.choice([True, False], size=(super_block_size, super_block_size), p=[0.8, 0.2])
                dense_block[~mask] = 0.0
                
                block_data = dense_block[dense_block != 0.0]
                block_rows, block_cols = np.where(dense_block != 0.0)
                
                global_rows = row_start + block_rows
                global_cols = col_start + block_cols
                
                data.extend(block_data.tolist())
                row_ind.extend(global_rows.tolist())
                col_ind.extend(global_cols.tolist())

            else:
                if np.random.rand() < low_rank_block_ratio:
                    U = np.random.randn(super_block_size, rank)
                    V = np.random.randn(super_block_size, rank)
                    low_rank_block = U @ V.T
                    
                    mask = np.random.choice([True, False], size=(super_block_size, super_block_size), p=[0.1, 0.9])
                    low_rank_block[~mask] = 0.0
                    
                    block_data = low_rank_block[low_rank_block != 0.0]
                    block_rows, block_cols = np.where(low_rank_block != 0.0)
                    
                    global_rows = row_start + block_rows
                    global_cols = col_start + block_cols
                    
                    data.extend(block_data.tolist())
                    row_ind.extend(global_rows.tolist())
                    col_ind.extend(global_cols.tolist())

    csr_mat = sp.csr_matrix((data, (row_ind, col_ind)), shape=(size, size))
    print(f"Generated FEM matrix: {size}x{size}, NNZ: {csr_mat.nnz}, Density: {csr_mat.nnz/(size*size):.2%}")
    return csr_mat

def save_csr_to_binary(csr_mat: sp.csr_matrix, output_dir: str = "data/fem/"):
    os.makedirs(output_dir, exist_ok=True)

    val_path = os.path.join(output_dir, "A_val.bin")
    with open(val_path, "wb") as f:
        val_data = csr_mat.data.astype(np.float32)
        f.write(val_data.tobytes())
    print(f"Saved {val_path}, Size: {os.path.getsize(val_path)/1024/1024:.2f} MB")

    row_ptr_path = os.path.join(output_dir, "A_row_ptr.bin")
    with open(row_ptr_path, "wb") as f:
        row_ptr_data = csr_mat.indptr.astype(np.int32)
        f.write(row_ptr_data.tobytes())
    print(f"Saved {row_ptr_path}, Size: {os.path.getsize(row_ptr_path)/1024/1024:.2f} MB")

    col_idx_path = os.path.join(output_dir, "A_col_idx.bin")
    with open(col_idx_path, "wb") as f:
        col_idx_data = csr_mat.indices.astype(np.int32)
        f.write(col_idx_data.tobytes())
    print(f"Saved {col_idx_path}, Size: {os.path.getsize(col_idx_path)/1024/1024:.2f} MB")

if __name__ == "__main__":
    fem_csr = generate_fem_stiffness_matrix(
        size=4096,
        non_zero_ratio=0.15,
        low_rank_block_ratio=0.9,
        super_block_size=64,
        rank=24
    )
    
    save_csr_to_binary(fem_csr, output_dir="data/fem/")
    print("="*50)
    print("FEM 4096x4096 dataset generation complete!")
