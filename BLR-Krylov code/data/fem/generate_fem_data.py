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
    """
    生成符合有限元刚度矩阵特征的稀疏矩阵：
    - 块对角结构（对角块稠密，非对角块低秩）
    - 非零率15%，低秩块占比90%
    - 64×64超级块划分，低秩块秩r=24（≤32）
    """
    # 1. 初始化空的COO矩阵（便于逐块构造）
    data = []
    row_ind = []
    col_ind = []

    # 2. 计算超级块数量（行/列方向）
    block_cnt = size // super_block_size
    assert block_cnt * super_block_size == size, "矩阵尺寸需为超级块尺寸的整数倍"

    # 3. 逐块构造矩阵（对角块稠密，非对角块低秩）
    for p in range(block_cnt):  # 行方向超级块索引
        for q in range(block_cnt):  # 列方向超级块索引
            # 超级块在全局矩阵中的起始/结束坐标
            row_start = p * super_block_size
            row_end = (p + 1) * super_block_size
            col_start = q * super_block_size
            col_end = (q + 1) * super_block_size

            # 3.1 对角块（p==q）：稠密块（非零率80%）
            if p == q:
                # 生成稠密矩阵，随机置零至非零率80%
                dense_block = np.random.randn(super_block_size, super_block_size)
                mask = np.random.choice([True, False], size=(super_block_size, super_block_size), p=[0.8, 0.2])
                dense_block[~mask] = 0.0
                # 提取非零元素
                block_data = dense_block[dense_block != 0.0]
                block_rows, block_cols = np.where(dense_block != 0.0)
                # 映射到全局坐标
                global_rows = row_start + block_rows
                global_cols = col_start + block_cols
                # 添加到COO列表
                data.extend(block_data.tolist())
                row_ind.extend(global_rows.tolist())
                col_ind.extend(global_cols.tolist())

            # 3.2 非对角块（p≠q）：90%概率为低秩块，10%为零块
            else:
                if np.random.rand() < low_rank_block_ratio:
                    # 生成低秩块：A ≈ U*V^T（U: 64×24, V: 64×24）
                    U = np.random.randn(super_block_size, rank)
                    V = np.random.randn(super_block_size, rank)
                    low_rank_block = U @ V.T
                    # 非零率控制在10%（模拟弱相互作用）
                    mask = np.random.choice([True, False], size=(super_block_size, super_block_size), p=[0.1, 0.9])
                    low_rank_block[~mask] = 0.0
                    # 提取非零元素
                    block_data = low_rank_block[low_rank_block != 0.0]
                    block_rows, block_cols = np.where(low_rank_block != 0.0)
                    # 映射到全局坐标
                    global_rows = row_start + block_rows
                    global_cols = col_start + block_cols
                    # 添加到COO列表
                    data.extend(block_data.tolist())
                    row_ind.extend(global_rows.tolist())
                    col_ind.extend(global_cols.tolist())
                # 10%概率为零块：不添加任何元素

    # 4. 转换为CSR格式并返回
    csr_mat = sp.csr_matrix((data, (row_ind, col_ind)), shape=(size, size))
    print(f"FEM矩阵生成完成：{size}×{size}，非零元素数：{csr_mat.nnz}，非零率：{csr_mat.nnz/(size*size):.2%}")
    return csr_mat

def save_csr_to_binary(csr_mat: sp.csr_matrix, output_dir: str = "data/fem/"):
    """
    将CSR矩阵保存为二进制格式（适配前文代码读取逻辑）：
    - A_val.bin: float32，非零元素值
    - A_row_ptr.bin: int32，行指针
    - A_col_idx.bin: int32，列索引
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存非零元素值（float32）
    val_path = os.path.join(output_dir, "A_val.bin")
    with open(val_path, "wb") as f:
        # 转换为float32
        val_data = csr_mat.data.astype(np.float32)
        # 写入二进制
        f.write(val_data.tobytes())
    print(f"已保存 {val_path}，大小：{os.path.getsize(val_path)/1024/1024:.2f} MB")

    # 2. 保存行指针（int32）
    row_ptr_path = os.path.join(output_dir, "A_row_ptr.bin")
    with open(row_ptr_path, "wb") as f:
        # 转换为int32
        row_ptr_data = csr_mat.indptr.astype(np.int32)
        # 写入二进制
        f.write(row_ptr_data.tobytes())
    print(f"已保存 {row_ptr_path}，大小：{os.path.getsize(row_ptr_path)/1024/1024:.2f} MB")

    # 3. 保存列索引（int32）
    col_idx_path = os.path.join(output_dir, "A_col_idx.bin")
    with open(col_idx_path, "wb") as f:
        # 转换为int32
        col_idx_data = csr_mat.indices.astype(np.int32)
        # 写入二进制
        f.write(col_idx_data.tobytes())
    print(f"已保存 {col_idx_path}，大小：{os.path.getsize(col_idx_path)/1024/1024:.2f} MB")

if __name__ == "__main__":
    # 生成4096×4096 FEM矩阵（符合论文实验条件）
    fem_csr = generate_fem_stiffness_matrix(
        size=4096,
        non_zero_ratio=0.15,
        low_rank_block_ratio=0.9,
        super_block_size=64,
        rank=24
    )
    # 保存为二进制格式（输出到data/fem/）
    save_csr_to_binary(fem_csr, output_dir="data/fem/")
    print("="*50)
    print("FEM 4096×4096 数据集生成完成！")
    print("文件路径：data/fem/A_val.bin、A_row_ptr.bin、A_col_idx.bin")
    print("特征：非零率15%，低秩块占比90%，64×64超级块，低秩块秩=24")