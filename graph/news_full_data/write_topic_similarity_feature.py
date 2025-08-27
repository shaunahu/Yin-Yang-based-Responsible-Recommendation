# topic similarity from news_topic_similarity_normalized.txt

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_GRAPH_DIR = "news_full_dataset_graph_data"
FEAT_FILE = "edge_feats.npy"
META_FILE = "features_meta.json"
FEATURE_NAME = "topic_similarity"
NUM_COLS_DEFAULT = 4
DTYPE = np.float16

TOPIC_SIM_TXT = os.path.join("edge_data", "news_topic_similarity_normalized.txt")  # 3 cols: t1 t2 sim

def ensure_meta_and_reserve_col(data_dir: str, feature_name: str, num_cols: int = NUM_COLS_DEFAULT):
    meta_path = os.path.join(data_dir, META_FILE)
    feats_path = os.path.join(data_dir, FEAT_FILE)

    nodes = np.load(os.path.join(data_dir, "nodes.npy"), allow_pickle=True)
    n = len(nodes)
    m = n * (n - 1) // 2
    print(f"[INFO] Graph loaded: {n} nodes, {m} edges")

    # Create feat matrix if missing
    if not os.path.exists(feats_path):
        print(f"[INFO] Creating new feature file at {feats_path}")
        feat_mm = np.memmap(feats_path, dtype=DTYPE, mode="w+", shape=(m, num_cols))
        feat_mm[:] = np.nan
        feat_mm.flush()

    # Load or init metadata
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        print(f"[INFO] Loaded existing meta: {meta}")
        if "num_cols" not in meta:
            meta["num_cols"] = num_cols
        if "features" not in meta:
            meta["features"] = {}
    else:
        meta = {"num_cols": num_cols, "dtype": str(DTYPE), "features": {}}
        print(f"[INFO] Creating new meta")

    # Assign/Reuse a column
    feats = meta["features"]
    if feature_name in feats:
        col = feats[feature_name]
        print(f"[INFO] Feature '{feature_name}' already assigned to column {col}")
    else:
        used = set(feats.values())
        free = [c for c in range(meta["num_cols"]) if c not in used]
        if not free:
            raise RuntimeError("edge_feats.npy has no free columns")
        col = free[0]
        feats[feature_name] = col
        with open(meta_path, "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Assigned feature '{feature_name}' to new column {col}")

    return n, m, col, meta

def load_topics_aligned_to_nodes(base_dir: str, nodes: np.ndarray) -> np.ndarray:
    items_csv = os.path.join(base_dir, "data_news", "items_filtered.csv")
    df = pd.read_csv(items_csv, dtype=str)
    id2row = {iid: i for i, iid in enumerate(df["item_id"].tolist())}
    topics = df["topic"].fillna("").str.strip().str.lower().tolist()
    per_node_topics = np.array([topics[id2row[iid]] for iid in nodes], dtype=object)
    print(f"[INFO] Loaded {len(per_node_topics)} node topics aligned to nodes.npy")
    return per_node_topics

def build_topic_sim_matrix(topic_pairs_path: str, known_topics: np.ndarray):
    uniq_topics = sorted(set(t for t in known_topics.tolist()))
    t2idx = {t: i for i, t in enumerate(uniq_topics)}
    K = len(uniq_topics)
    print(f"[INFO] Building topic similarity matrix for {K} unique topics")

    M = np.full((K, K), np.nan, dtype=np.float32)
    with open(topic_pairs_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            t1, t2, val = parts[0].lower(), parts[1].lower(), parts[2]
            try:
                sim = float(val)
            except:
                continue
            if t1 in t2idx and t2 in t2idx:
                i, j = t2idx[t1], t2idx[t2]
                M[i, j] = sim
                M[j, i] = sim

    # Fill diagonal
    for i in range(K):
        if np.isnan(M[i, i]):
            M[i, i] = 1.0

    missing_mask = np.isnan(M)
    np.fill_diagonal(missing_mask, False)
    num_missing = int(missing_mask.sum())
    if num_missing > 0:
        M[missing_mask] = 0.0
        print(f"[WARN] {num_missing} topic-topic pairs missing, filled with 0.0")

    return M, t2idx

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, BASE_GRAPH_DIR)

    # Load nodes
    nodes = np.load(os.path.join(data_dir, "nodes.npy"), allow_pickle=True)
    n = len(nodes)

    # Load per-node topics
    per_node_topics = load_topics_aligned_to_nodes(BASE_DIR, nodes)

    # Build topic sim matrix
    topic_pairs_path = os.path.join(BASE_DIR, TOPIC_SIM_TXT)
    M, t2idx = build_topic_sim_matrix(topic_pairs_path, per_node_topics)

    # Map topics to indices
    topic_idx = np.array([t2idx.get(t, -1) for t in per_node_topics], dtype=np.int32)
    if (topic_idx < 0).any():
        num_unknown = int((topic_idx < 0).sum())
        raise ValueError(f"{num_unknown} node topics not found in topic similarity file {topic_pairs_path}")

    # Open memmap
    _, m, col, meta = ensure_meta_and_reserve_col(data_dir, FEATURE_NAME, num_cols=NUM_COLS_DEFAULT)
    feat_mm = np.memmap(os.path.join(data_dir, FEAT_FILE),
                        dtype=DTYPE, mode="r+",
                        shape=(m, meta["num_cols"]))

    print(f"[INFO] Writing feature '{FEATURE_NAME}' into column {col}")

    # Row-wise writing
    cursor = 0
    for i in tqdm(range(n - 1), desc=f"Writing {FEATURE_NAME}"):
        cnt = n - i - 1
        if cnt <= 0:
            continue
        ti = topic_idx[i]
        tj = topic_idx[i + 1 : n]
        sims = M[ti, tj]
        feat_mm[cursor : cursor + cnt, col] = sims.astype(DTYPE, copy=False)
        cursor += cnt

        if i % 5000 == 0 and i > 0:
            print(f"[DEBUG] Processed row {i}/{n-1}, cursor={cursor}")

    feat_mm.flush()
    print(f"[INFO] Finished writing {FEATURE_NAME}. Column {col} updated in {FEAT_FILE}")

if __name__ == "__main__":
    main()
