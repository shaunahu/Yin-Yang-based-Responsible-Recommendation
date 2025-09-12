# semantic_similarity.py
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer

# ---- locations relative to this script (features/) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data_news"))
GRAPH_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "news_full_dataset_graph_data"))

FEAT_FILE = "edge_feats.npy"
META_FILE = "features_meta.json"

FEATURE_NAME = "semantic_similarity_ta"  # (title+abstract semantic similarity)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NUM_COLS_DEFAULT = 4
DTYPE = np.float16  # on-disk dtype to keep file small
BATCH_SIZE = 512    # adjust for your GPU/CPU memory

def ensure_meta_and_reserve_col(data_dir: str, feature_name: str, num_cols: int = NUM_COLS_DEFAULT):
    """Create/open edge_feats.npy and features_meta.json, and reserve a column for this feature."""
    meta_path = os.path.join(data_dir, META_FILE)
    feats_path = os.path.join(data_dir, FEAT_FILE)

    nodes = np.load(os.path.join(data_dir, "nodes.npy"), allow_pickle=True)
    n = len(nodes)
    m = n * (n - 1) // 2

    # Create features file if missing
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

    # Assign/Reuse a column
    feats = meta["features"]
    if feature_name in feats:
        col = feats[feature_name]
    else:
        used = set(feats.values())
        for c in range(meta["num_cols"]):
            if c not in used:
                feats[feature_name] = c
                col = c
                break
        else:
            raise RuntimeError("edge_feats.npy columns are full. Expand columns or use separate files.")
        with open(meta_path, "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    return n, m, col, meta

def main():
    # Resolve paths
    items_csv = os.path.join(DATA_DIR, "items_filtered.csv")
    nodes_path = os.path.join(GRAPH_DIR, "nodes.npy")
    feats_path = os.path.join(GRAPH_DIR, FEAT_FILE)

    print(f"[INFO] DATA_DIR  = {DATA_DIR}")
    print(f"[INFO] GRAPH_DIR = {GRAPH_DIR}")

    # Load node order (item_id sequence used for edges)
    nodes = np.load(nodes_path, allow_pickle=True)
    n = len(nodes)
    m = n * (n - 1) // 2
    print(f"[INFO] nodes={n:,}, edges={m:,}")

    # Read items and align texts to nodes order
    df = pd.read_csv(items_csv, dtype=str)
    id2row = {iid: idx for idx, iid in enumerate(df["item_id"].tolist())}

    titles = df["title"].fillna("").tolist()
    abstracts = df["abstract"].fillna("").tolist()
    texts_all = [titles[id2row[iid]] + " " + abstracts[id2row[iid]] for iid in nodes]

    # Load model (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading model: {MODEL_NAME} on {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # Encode all texts as unit vectors (normalize_embeddings=True => cosine == dot)
    print("[INFO] Encoding texts -> embeddings")
    with torch.inference_mode():
        emb = model.encode(
            texts_all,
            batch_size=BATCH_SIZE,
            convert_to_tensor=True,
            device=device,
            normalize_embeddings=True  # L2 normalize; cosine == dot
        )  # (n, d), float32 on device

    # Prepare feature column in memmap
    _, _, col, meta = ensure_meta_and_reserve_col(GRAPH_DIR, FEATURE_NAME, num_cols=NUM_COLS_DEFAULT)
    feat_mm = np.memmap(feats_path, dtype=DTYPE, mode="r+", shape=(m, meta["num_cols"]))

    # Row-wise write: for each i, compute sim(i, i+1..n-1) = (dot + 1)/2
    cursor = 0
    print(f"[INFO] Writing feature '{FEATURE_NAME}' -> column {col}")
    with torch.inference_mode(), tqdm(total=n - 1, desc="semantic sims (row-wise)") as pbar:
        for i in range(n - 1):
            cnt = n - i - 1
            if cnt <= 0:
                pbar.update(1); continue

            vi = emb[i].unsqueeze(0)        # (1, d)
            Vj = emb[i + 1 : n]             # (cnt, d)

            sims = torch.matmul(Vj, vi.T).squeeze(1)  # cosine since normalized ∈ [-1,1]
            sims = (sims + 1.0) * 0.5                  # map to [0,1]

            sims_np = sims.detach().to("cpu").numpy().astype(DTYPE, copy=False)
            feat_mm[cursor : cursor + cnt, col] = sims_np

            cursor += cnt
            pbar.update(1)
