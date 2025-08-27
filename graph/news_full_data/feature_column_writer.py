# feature_column_writer.py
import os, json, numpy as np
from tqdm import tqdm
import pandas as pd

BASE_GRAPH_DIR = "news_full_dataset_graph_data"
FEAT_FILE = "edge_feats.npy"
META_FILE = "features_meta.json"
NUM_COLS = 4
DTYPE = np.float16  # 统一 dtype；不同特征若需不同精度，可用独立文件方案（见下）

def _paths(base_dir):
    data_dir = os.path.join(base_dir, BASE_GRAPH_DIR)
    return {
        "dir": data_dir,
        "nodes": os.path.join(data_dir, "nodes.npy"),
        "src": os.path.join(data_dir, "edges_src_int32.npy"),
        "dst": os.path.join(data_dir, "edges_dst_int32.npy"),
        "feat": os.path.join(data_dir, FEAT_FILE),
        "meta": os.path.join(data_dir, META_FILE),
    }

def _ensure_feat_and_meta(paths):
    nodes = np.load(paths["nodes"], allow_pickle=True)
    n = len(nodes)
    m = n * (n - 1) // 2
    # 建立/打开特征 memmap
    if not os.path.exists(paths["feat"]):
        feat_mm = np.memmap(paths["feat"], dtype=DTYPE, mode="w+", shape=(m, NUM_COLS))
        feat_mm[:] = np.nan  # 用 NaN 表示“未写入”
        feat_mm.flush()
    # 建立/读取元数据
    if os.path.exists(paths["meta"]):
        with open(paths["meta"], "r") as f:
            meta = json.load(f)
    else:
        meta = {"num_cols": NUM_COLS, "dtype": str(DTYPE), "features": {}}  # 特征名->列号
        with open(paths["meta"], "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    return n, m, meta

def _reserve_column(meta, feature_name):
    """为 feature_name 分配列；若已存在，则复用；否则分配空列。"""
    feats = meta["features"]
    if feature_name in feats:
        return feats[feature_name]
    used = set(feats.values())
    for col in range(meta["num_cols"]):
        if col not in used:
            feats[feature_name] = col
            return col
    raise RuntimeError("edge_feats.npy 列已用满，请扩容或采用独立文件方案。")

def _save_meta(paths, meta):
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def write_feature_from_rowwise_diff(base_dir, feature_name, per_node_values, batch_rows=2000):
    """
    按上三角顺序顺序写入：给定每个节点的标量值 v[i]，写入 |v[i]-v[j]| 到该特征列。
    per_node_values: np.ndarray shape=(n,), dtype float
    """
    paths = _paths(base_dir)
    n, m, meta = _ensure_feat_and_meta(paths)
    if len(per_node_values) != n:
        raise ValueError(f"per_node_values 长度({len(per_node_values)})必须等于 nodes 数量({n})。")

    col = _reserve_column(meta, feature_name)
    feat_mm = np.memmap(paths["feat"], dtype=DTYPE, mode="r+", shape=(m, meta["num_cols"]))
    cursor = 0
    with tqdm(total=n-1, desc=f"writing feature '{feature_name}' -> col {col}") as pbar:
        for i_block_start in range(0, n-1, batch_rows):
            i_block_end = min(n-1, i_block_start + batch_rows)
            for i in range(i_block_start, i_block_end):
                cnt = n - i - 1
                if cnt <= 0:
                    continue
                diff = np.abs(per_node_values[i] - per_node_values[i+1:n]).astype(DTYPE)
                feat_mm[cursor:cursor+cnt, col] = diff
                cursor += cnt
                pbar.update(1)
    feat_mm.flush()
    _save_meta(paths, meta)
    print(f"✅ 写入完成：{feature_name} -> col {col}")

def upper_tri_index(n, i, j):
    """将 (i<j) 映射到线性索引 k（上三角行优先）。"""
    k_before_i = (n * (n - 1) // 2) - ((n - 1 - i) * (n - i) // 2)
    return k_before_i + (j - i - 1)

def write_feature_from_pairs_txt(base_dir, feature_name, txt_path, sep="\t", batch_print=500000):
    """
    从形如 `item_i<sep>item_j<sep>value` 的文件写入（随机寻址，较慢）。
    """
    paths = _paths(base_dir)
    nodes = np.load(paths["nodes"], allow_pickle=True)
    n, m, meta = _ensure_feat_and_meta(paths)
    id2idx = {iid: idx for idx, iid in enumerate(nodes.tolist())}
    col = _reserve_column(meta, feature_name)
    feat_mm = np.memmap(paths["feat"], dtype=DTYPE, mode="r+", shape=(m, meta["num_cols"]))

    total = sum(1 for _ in open(txt_path, "r"))
    with open(txt_path, "r") as f, tqdm(total=total, desc=f"backfilling '{feature_name}' -> col {col}") as pbar:
        for t, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                pbar.update(1); continue
            a, b, val = line.split(sep)
            if a not in id2idx or b not in id2idx:
                pbar.update(1); continue
            i, j = id2idx[a], id2idx[b]
            if i == j:
                pbar.update(1); continue
            if i > j:
                i, j = j, i
            k = upper_tri_index(n, i, j)
            feat_mm[k, col] = np.array(val, dtype=DTYPE)
            pbar.update(1)
            if t % batch_print == 0:
                feat_mm.flush()
    feat_mm.flush()
    _save_meta(paths, meta)
    print(f"✅ 回填完成：{feature_name} -> col {col}")
