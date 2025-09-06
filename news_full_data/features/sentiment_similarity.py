# write_sentiment_similarity_feature.py
import os
import json
import numpy as np
import pandas as pd
from textblob import TextBlob
from tqdm import tqdm

# ---- locations relative to this script (features/) ----
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", "data_news"))
GRAPH_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "news_full_dataset_graph_data"))

FEAT_FILE = "edge_feats.npy"
META_FILE = "features_meta.json"
FEATURE_NAME = "sentiment_similarity"   # name recorded in features_meta.json
DTYPE = np.float16
BATCH_ROWS = 2000
NUM_COLS_DEFAULT = 4

def ensure_meta_and_reserve_col(graph_dir: str, feature_name: str, num_cols: int = NUM_COLS_DEFAULT):
    """Create/open edge_feats.npy and features_meta.json, and reserve/reuse a column for this feature."""
    meta_path  = os.path.join(graph_dir, META_FILE)
    feats_path = os.path.join(graph_dir, FEAT_FILE)

    # Determine number of edges from nodes.npy
    nodes = np.load(os.path.join(graph_dir, "nodes.npy"), allow_pickle=True)
    n = len(nodes)
    m = n * (n - 1) // 2

    # Create feature file if missing (filled with NaN)
    if not os.path.exists(feats_path):
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

    # Assign/reuse a column
    feats = meta["features"]
    if feature_name in feats:
        col = feats[feature_name]
    else:
        used = set(feats.values())
        free = next((c for c in range(meta["num_cols"]) if c not in used), None)
        if free is None:
            raise RuntimeError("edge_feats.npy has no free columns; expand or use per-feature files.")
        col = free
        feats[feature_name] = col
        with open(meta_path, "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    return n, m, col, meta

def main():
    print(f"[INFO] DATA_DIR  = {DATA_DIR}")
    print(f"[INFO] GRAPH_DIR = {GRAPH_DIR}")

    # Load node order (item_id sequence used for edges)
    nodes = np.load(os.path.join(GRAPH_DIR, "nodes.npy"), allow_pickle=True)
    n = len(nodes)

    # Load items and compute per-item sentiment for title and abstract
    items_csv = os.path.join(DATA_DIR, "items_filtered.csv")
    df = pd.read_csv(items_csv, dtype=str)
    id2row = {iid: i for i, iid in enumerate(df["item_id"].tolist())}

    def norm_sent(txt: str) -> float:
        # TextBlob polarity in [-1,1] -> normalize to [0,1]
        return (TextBlob(txt).sentiment.polarity + 1.0) / 2.0

    titles    = df["title"].fillna("").tolist()
    abstracts = df["abstract"].fillna("").tolist()

    title_sent_all    = np.zeros(len(df), dtype=np.float32)
    abstract_sent_all = np.zeros(len(df), dtype=np.float32)

    for r in tqdm(range(len(df)), desc="per-item sentiment (title & abstract)"):
        title_sent_all[r]    = norm_sent(titles[r])
        abstract_sent_all[r] = norm_sent(abstracts[r])

    # Align to nodes.npy order
    title_sent    = np.array([title_sent_all[id2row[iid]]    for iid in nodes], dtype=np.float32)
    abstract_sent = np.array([abstract_sent_all[id2row[iid]] for iid in nodes], dtype=np.float32)

    # Prepare memmap and column
    n, m, col, meta = ensure_meta_and_reserve_col(GRAPH_DIR, FEATURE_NAME, num_cols=NUM_COLS_DEFAULT)
    feat_mm = np.memmap(os.path.join(GRAPH_DIR, FEAT_FILE), dtype=DTYPE, mode="r+", shape=(m, meta["num_cols"]))

    # Write in upper-triangular order: sim = 1 - (|Δtitle| + |Δabstract|)/2  -> in [0,1]
    cursor = 0
    with tqdm(total=n - 1, desc=f"writing '{FEATURE_NAME}' -> col {col}") as pbar:
        for i_block_start in range(0, n - 1, BATCH_ROWS):
            i_block_end = min(n - 1, i_block_start + BATCH_ROWS)
            for i in range(i_block_start, i_block_end):
                cnt = n - i - 1
                if cnt <= 0:
                    pbar.update(1); continue
                d_title = np.abs(title_sent[i]    - title_sent[i + 1 : n])   # (cnt,)
                d_abs   = np.abs(abstract_sent[i] - abstract_sent[i + 1 : n])# (cnt,)
                sim = 1.0 - (d_title + d_abs) / 2.0                           # (cnt,)
                feat_mm[cursor : cursor + cnt, col] = sim.astype(DTYPE)
                cursor += cnt
                pbar.update(1)

    feat_mm.flush()
    print(f"✅ Done. Sentiment similarity written to {os.path.join(GRAPH_DIR, FEAT_FILE)} (col {col}).")

if __name__ == "__main__":
    main()
