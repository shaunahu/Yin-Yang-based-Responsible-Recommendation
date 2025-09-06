import os
import json
import numpy as np
import pandas as pd
from itertools import combinations
from tqdm import tqdm
from collections import Counter

# Resolve base dirs relative to this script
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", "news_full_dataset_graph_data"))
BEHAVIORS_CSV = os.path.abspath(os.path.join(BASE_DIR, "..", "data_news", "new_behaviors_filtered.csv"))

FEATURE_NAME = "impression_cooccurrence_prob"
DTYPE = np.float16
NUM_COLS_DEFAULT = 4
CHUNK_SIZE = 100_000  # adjust for your memory

def ensure_meta_and_reserve_col(graph_dir: str, feature_name: str, num_cols: int = NUM_COLS_DEFAULT):
    meta_path = os.path.join(graph_dir, "features_meta.json")
    feats_path = os.path.join(graph_dir, "edge_feats.npy")

    nodes = np.load(os.path.join(graph_dir, "nodes.npy"), allow_pickle=True)
    n = len(nodes); m = n * (n - 1) // 2
    print(f"[INFO] Graph: {n} nodes, {m} edges")

    # Create feature file if missing
    if not os.path.exists(feats_path):
        print(f"[INFO] Creating feature memmap at: {feats_path}")
        feat_mm = np.memmap(feats_path, dtype=DTYPE, mode="w+", shape=(m, num_cols))
        feat_mm[:] = np.nan
        feat_mm.flush()

    # Load or initialize meta
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta.setdefault("num_cols", num_cols)
        meta.setdefault("features", {})
    else:
        meta = {"num_cols": num_cols, "dtype": str(DTYPE), "features": {}}

    # Assign/reuse column
    feats = meta["features"]
    if feature_name in feats:
        col = feats[feature_name]
        print(f"[INFO] Feature '{feature_name}' -> existing column {col}")
    else:
        used = set(feats.values())
        free = [c for c in range(meta["num_cols"]) if c not in used]
        if not free:
            raise RuntimeError("No free columns in edge_feats.npy. Expand or use per-feature files.")
        col = free[0]
        feats[feature_name] = col
        with open(meta_path, "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Feature '{feature_name}' -> new column {col}")

    return n, m, col, meta

def parse_impression(cell: str):
    if not isinstance(cell, str) or not cell.strip():
        return []
    toks = cell.replace(",", " ").split()
    return sorted(set(t.split("-", 1)[0] for t in toks if t))  # unique per row

def count_pair_frequencies(csv_path: str, chunk_size: int = CHUNK_SIZE):
    """Return (pair_counter, total_nonempty_rows). pair keys are (idA,idB) with idA<idB."""
    pair_counts = Counter()
    total_rows = 0

    print(f"[INFO] Scanning impressions from: {csv_path}")
    reader = pd.read_csv(csv_path, dtype=str, chunksize=chunk_size)
    for chunk in tqdm(reader, desc="Counting co-occurrences"):
        if "impression" not in chunk.columns:
            lower = {c.lower(): c for c in chunk.columns}
            if "impression" in lower:
                chunk = chunk.rename(columns={lower["impression"]: "impression"})
            else:
                raise ValueError("Column 'impression' not found in behaviors CSV.")

        for cell in chunk["impression"].fillna(""):
            items = parse_impression(cell)
            if not items:
                continue
            total_rows += 1
            if len(items) < 2:
                continue
            for a, b in combinations(items, 2):
                pair_counts[(a, b)] += 1

    print(f"[INFO] Counted {total_rows} non-empty impressions, {len(pair_counts):,} unique pairs")
    return pair_counts, total_rows

def write_prob_feature(graph_dir: str, feature_name: str, pair_counts: Counter, total_rows: int, col: int, num_cols: int):
    nodes = np.load(os.path.join(graph_dir, "nodes.npy"), allow_pickle=True)
    n = len(nodes); m = n * (n - 1) // 2
    nodes = nodes.tolist()
    feat_mm = np.memmap(os.path.join(graph_dir, "edge_feats.npy"),
                        dtype=DTYPE, mode="r+", shape=(m, num_cols))

    get_prob = lambda a, b: pair_counts.get((a, b), pair_counts.get((b, a), 0)) / total_rows

    cursor = 0
    print(f"[INFO] Writing '{feature_name}' to column {col}")
    for i in tqdm(range(n - 1), desc=f"Writing {feature_name}"):
        a = nodes[i]
        cnt = n - i - 1
        if cnt <= 0:
            continue
        probs = np.empty((cnt,), dtype=DTYPE)
        for off, j in enumerate(range(i + 1, n)):
            b = nodes[j]
            probs[off] = get_prob(a, b)
        feat_mm[cursor: cursor + cnt, col] = probs
        cursor += cnt

    feat_mm.flush()
    print(f"[INFO] Done. Column {col} updated.")

def main():
    pair_counts, total_rows = count_pair_frequencies(BEHAVIORS_CSV)
    n, m, col, meta = ensure_meta_and_reserve_col(GRAPH_DIR, FEATURE_NAME, num_cols=NUM_COLS_DEFAULT)
    write_prob_feature(GRAPH_DIR, FEATURE_NAME, pair_counts, total_rows, col, meta["num_cols"])

if __name__ == "__main__":
    main()
