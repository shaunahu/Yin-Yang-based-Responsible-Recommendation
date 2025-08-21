import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# ========= config =========
ID_COL = "item_id"
FEATURE_DTYPE = np.float16   # can be changed to np.uint8 / np.float32
OUT_DIR_NAME = "news_full_dataset_graph_data"
SRC_FILENAME = "edges_src_int32.npy"
DST_FILENAME = "edges_dst_int32.npy"
FEAT_FILENAME = "edge_feats.npy"     # shape: (m, 4)
BATCH_ROWS = 2000                    # Use i as the row block size, adjust as needed (the larger the value, the faster it is but the more memory it takes up)

def compute_edge_features_batch(node_ids_left: np.ndarray,
                                node_ids_right: np.ndarray) -> np.ndarray:
    """
    你要实现的边特征计算函数（批量版）。
    参数：
      node_ids_left:  shape (cnt,)   左端节点的“全局索引 i”或 ID
      node_ids_right: shape (cnt,)   右端节点的“全局索引 j”或 ID
    返回：
      feats: shape (cnt, 4), dtype = FEATURE_DTYPE
    """
    # === sample） ===
    feats = np.zeros((len(node_ids_left), 4), dtype=FEATURE_DTYPE)
    return feats

def main():
    # 目录
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data_news")
    OUT_DIR  = os.path.join(BASE_DIR, OUT_DIR_NAME)
    if not os.path.isdir(OUT_DIR):
        raise FileNotFoundError(f"Output directory not found: {OUT_DIR}")

    # read item
    csv_path = os.path.join(DATA_DIR, "items_filtered.csv")
    df = pd.read_csv(csv_path, dtype=str)
    if ID_COL not in df.columns:
        raise ValueError(f"Expected column '{ID_COL}' in {csv_path}. Found: {list(df.columns)}")

    nodes = df[ID_COL].dropna().unique().tolist()
    n = len(nodes)
    m = n * (n - 1) // 2
    print(f"nodes={n:,}, edges={m:,}")

    # Save node order (for easy retrieval)
    nodes_npy = os.path.join(OUT_DIR, "nodes.npy")
    np.save(nodes_npy, np.array(nodes, dtype=object))

    # create memmap file
    src_path = os.path.join(OUT_DIR, SRC_FILENAME)
    dst_path = os.path.join(OUT_DIR, DST_FILENAME)
    feat_path = os.path.join(OUT_DIR, FEAT_FILENAME)

    src_mm  = np.memmap(src_path,  dtype=np.int32,   mode="w+", shape=(m,))
    dst_mm  = np.memmap(dst_path,  dtype=np.int32,   mode="w+", shape=(m,))
    feat_mm = np.memmap(feat_path, dtype=FEATURE_DTYPE, mode="w+", shape=(m, 4))

    # Write the upper triangle block by block (i < j)
    cursor = 0
    with tqdm(total=m, desc="Writing edges & feats") as pbar:
        # Batched into "row chunks" (i from 0..n-2)
        for i_block_start in range(0, n - 1, BATCH_ROWS):
            i_block_end = min(n - 1, i_block_start + BATCH_ROWS)  # i 取到 n-2
            # For each i, j from i+1..n-1
            for i in range(i_block_start, i_block_end):
                j_count = n - (i + 1)
                if j_count <= 0:
                    continue

                # The j index of the current batch
                j_idx = np.arange(i + 1, n, dtype=np.int32)

                # 写 src/dst
                src_mm[cursor: cursor + j_count] = i
                dst_mm[cursor: cursor + j_count] = j_idx

                # Calculate the features and write them (pass "index i and j_idx" to your feature function)
                feats = compute_edge_features_batch(
                    node_ids_left=np.full((j_count,), i, dtype=np.int32),
                    node_ids_right=j_idx
                )
                if feats.shape != (j_count, 4):
                    raise ValueError(f"compute_edge_features_batch must return shape (cnt, 4), got {feats.shape}")
                feat_mm[cursor: cursor + j_count, :] = feats

                cursor += j_count
                pbar.update(j_count)

    # Flush to disk
    src_mm.flush()
    dst_mm.flush()
    feat_mm.flush()

    print("Done.")
    print(f"src -> {src_path}")
    print(f"dst -> {dst_path}")
    print(f"feats -> {feat_path}")
    est_bytes_per_edge = 8 + (4 * np.dtype(FEATURE_DTYPE).itemsize)
    print(f"~Estimated disk size: {m * est_bytes_per_edge / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
