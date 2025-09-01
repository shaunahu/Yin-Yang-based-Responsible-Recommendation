# write_topic_similarity_feature.py
# Insert topic similarity (from topic–topic table) into edge_feats.npy

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- locations relative to this script (features/) ---
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.abspath(os.path.join(BASE_DIR, "..", "data_news"))
GRAPH_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", "news_full_dataset_graph_data"))
EDGE_DIR   = os.path.abspath(os.path.join(BASE_DIR, "..", "edge_data"))

FEAT_FILE = "edge_feats.npy"
META_FILE = "features_meta.json"
FEATURE_NAME = "topic_similarity"
NUM_COLS_DEFAULT = 4
DTYPE = np.float16

TOPIC_SIM_TXT = os.path.join(EDGE_DIR, "news_topic_similarity_normalized.txt")  # 3 cols: t1 t2 sim

def ensure_meta_and_reserve_col(graph_dir: str, feature_name: str, num_cols: int = NUM_COLS_DEFAULT):
    """Create/open edge_feats.npy and features_meta.json; reserve (or reuse) a column."""
    meta_path  = os.path.join(graph_dir, META_FILE)
    feats_path = os.path.join(graph_dir, FEAT_FILE)

    nodes = np.load(os.path.join(graph_dir, "nodes.npy"), allow_pickle=True)
    n = len(nodes)
    m = n * (n - 1) // 2
    print(f"[INFO] Graph loaded: {n} nodes, {m} edges")

    if not os.path.exists(feats_path):
        print(f"[INFO] Creating feature memmap at {feats_path}")
        feat_mm = np.memmap(feats_path, dtype=DTYPE, mode="w+", shape=(m, num_cols))
        feat_mm[:] = np.nan
        feat_mm.flush()

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta.setdefault("num_cols", num_cols)
        meta.setdefault("features", {})
        print(f"[INFO] Loaded meta: {meta}")
    else:
        meta = {"num_cols": num_cols, "dtype": str(DTYPE), "features": {}}
        print(f"[INFO] Creating new meta")

    feats = meta["features"]
    if feature_name in feats:
        col = feats[feature_name]
        print(f"[INFO] Feature '{feature_name}' already uses column {col}")
    else:
        used = set(feats.values())
        free = [c for c in range(meta["num_cols"]) if c not in used]
        if not free:
            raise RuntimeError("edge_feats.npy has no free columns; expand or use per-feature files.")
        col = free[0]
        feats[feature_name] = col
        with open(meta_path, "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Assigned feature '{feature_name}' to new column {col}")

    return n, m, col, meta

def load_topics_aligned_to_nodes(graph_dir: str, data_dir: str, nodes: np.ndarray) -> np.ndarray:
    """Return per-node topic strings aligned to nodes.npy order (lowercased)."""
    items_csv = os.path.join(data_dir, "items_filtered.csv")
    df = pd.read_csv(items_csv, dtype=str)
    id2row = {iid: i for i, iid in enumerate(df["item_id"].tolist())}
    topics = df["topic"].fillna("").str.strip().str.lower().tolist()
    per_node_topics = np.array([topics[id2row[iid]] for iid in nodes], dtype=object)
    print(f"[INFO] Loaded {len(per_node_topics)} node topics aligned to nodes.npy")
    return per_node_topics

def build_topic_sim_matrix(topic_pairs_path: str, known_topics: np.ndarray):
    """Build a dense K×K similarity matrix only for topics present on nodes."""
    uniq_topics = sorted(set(known_topics.tolist()))
    t2idx = {t: i for i, t in enumerate(uniq_topics)}
    K = len(uniq_topics)
    print(f"[INFO] Building topic similarity matrix for {K} unique topics")
    M = np.full((K, K), np.nan, dtype=np.float32)

    if not os.path.exists(topic_pairs_path):
        raise FileNotFoundError(f"Topic similarity file not found: {topic_pairs_path}")

    with open(topic_pairs_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            t1, t2, val = parts[0].lower(), parts[1].lower(), parts[2]
            try:
                sim = float(val)
            except ValueError:
                continue
            if t1 in t2idx and t2 in t2idx:
                i, j = t2idx[t1], t2idx[t2]
                M[i, j] = sim
                M[j, i] = sim  # assume symmetry

    # Fill diagonal with perfect self-similarity
    for i in range(K):
        if np.isnan(M[i, i]):
            M[i, i] = 1.0

    # Any unseen cross-topic pairs default to 0.0 (log a count)
    missing_mask = np.isnan(M)
    np.fill_diagonal(missing_mask, False)
    num_missing = int(missing_mask.sum())
    if num_missing > 0:
        M[missing_mask] = 0.0
        print(f"[WARN] {num_missing} topic pairs missing in file; filled with 0.0")

    return M, t2idx

def main():
    print(f"[INFO] DATA_DIR  = {DATA_DIR}")
    print(f"[INFO] GRAPH_DIR = {GRAPH_DIR}")
    print(f"[INFO] TOPIC_SIM = {TOPIC_SIM_TXT}")

    # Load nodes and per-node topics
    nodes = np.load(os.path.join(GRAPH_DIR, "nodes.npy"), allow_pickle=True)
    n = len(nodes)
    per_node_topics = load_topics_aligned_to_nodes(GRAPH_DIR, DATA_DIR, nodes)

    # Build topic similarity table over topics actually present
    M, t2idx = build_topic_sim_matrix(TOPIC_SIM_TXT, per_node_topics)

    # Map each node's topic to matrix indices
    topic_idx = np.array([t2idx.get(t, -1) for t in per_node_topics], dtype=np.int32)
    if (topic_idx < 0).any():
        num_unknown = int((topic_idx < 0).sum())
        raise ValueError(f"{num_unknown} node topics not found in topic similarity table.")

    # Reserve column and open memmap
    _, m, col, meta = ensure_meta_and_reserve_col(GRAPH_DIR, FEATURE_NAME, num_cols=NUM_COLS_DEFAULT)
    feat_mm = np.memmap(os.path.join(GRAPH_DIR, FEAT_FILE), dtype=DTYPE, mode="r+",
                        shape=(m, meta["num_cols"]))

    print(f"[INFO] Writing feature '{FEATURE_NAME}' to column {col}")
    # Upper-triangular row-wise write: edge (i, j) with i<j at contiguous positions
    cursor = 0
    for i in tqdm(range(n - 1), desc=f"Writing {FEATURE_NAME}"):
        cnt = n - i - 1
        if cnt <= 0:
            continue
        ti = topic_idx[i]
        tj = topic_idx[i + 1 : n]
        sims = M[ti, tj]  # (cnt,)
        feat_mm[cursor : cursor + cnt, col] = sims.astype(DTYPE, copy=False)
        cursor += cnt
        if i % 5000 == 0 and i > 0:
            print(f"[DEBUG] processed row {i}/{n-1}, cursor={cursor}")

    feat_mm.flush()
    print(f"✅ Done. Topic similarity written to {os.path.join(GRAPH_DIR, FEAT_FILE)} (col {col}).")

if __name__ == "__main__":
    main()
