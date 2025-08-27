import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer

BASE_GRAPH_DIR = "news_full_dataset_graph_data"
FEAT_FILE = "edge_feats.npy"
META_FILE = "features_meta.json"

FEATURE_NAME = "semantic_similarity_ta"  # (title+abstract semantic similarity)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NUM_COLS_DEFAULT = 4
DTYPE = np.float16  # stored dtype on disk (keeps file small)

def ensure_meta_and_reserve_col(data_dir: str, feature_name: str, num_cols: int = NUM_COLS_DEFAULT):
    """Create/open edge_feats.npy and features_meta.json, and reserve a column for this feature."""
    meta_path = os.path.join(data_dir, META_FILE)
    feats_path = os.path.join(data_dir, FEAT_FILE)

    nodes = np.load(os.path.join(data_dir, "nodes.npy"), allow_pickle=True)
    n = len(nodes)
    m = n * (n - 1) // 2

    # Create feat matrix if missing
    if not os.path.exists(feats_path):
        feat_mm = np.memmap(feats_path, dtype=DTYPE, mode="w+", shape=(m, num_cols))
        feat_mm[:] = np.nan
        feat_mm.flush()

    # Load or initialize meta
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        # If file exists but column count smaller than requested, keep existing
        if "num_cols" not in meta:
            meta["num_cols"] = num_cols
        if "features" not in meta:
            meta["features"] = {}
    else:
        meta = {"num_cols": num_cols, "dtype": str(DTYPE), "features": {}}

    # Assign/Reuse a column for this feature
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
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, BASE_GRAPH_DIR)

    # Load node order (item_id sequence used for edges)
    nodes = np.load(os.path.join(data_dir, "nodes.npy"), allow_pickle=True)
    n = len(nodes)
    m = n * (n - 1) // 2
    print(f"nodes={n:,}, edges={m:,}")

    # Read items and align texts to nodes order
    items_csv = os.path.join(BASE_DIR, "data_news", "items_filtered.csv")
    df = pd.read_csv(items_csv, dtype=str)
    id2row = {iid: idx for idx, iid in enumerate(df["item_id"].tolist())}

    # Build text per node (title + abstract); aligned strictly to nodes.npy
    titles = df["title"].fillna("").tolist()
    abstracts = df["abstract"].fillna("").tolist()
    texts_all = [titles[id2row[iid]] + " " + abstracts[id2row[iid]] for iid in nodes]

    # Load model (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {MODEL_NAME} on {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # Encode all texts as unit vectors (normalize_embeddings=True => cosine = dot)
    print("Encoding texts -> embeddings")
    with torch.inference_mode():
        emb = model.encode(
            texts_all,
            batch_size=512,            # adjust if you have more/less GPU memory
            convert_to_tensor=True,
            device=device,
            normalize_embeddings=True  # L2 normalize; cosine == dot
        )  # shape: (n, d), torch.float32 on device

    # Prepare feature column in memmap
    _, _, col, meta = ensure_meta_and_reserve_col(data_dir, FEATURE_NAME, num_cols=NUM_COLS_DEFAULT)
    feat_mm = np.memmap(
        os.path.join(data_dir, FEAT_FILE),
        dtype=DTYPE, mode="r+",
        shape=(m, meta["num_cols"])
    )

    # Row-wise write: for each i, compute sim(i, i+1..n-1) = (dot + 1)/2
    cursor = 0
    print(f"Writing feature '{FEATURE_NAME}' -> col {col}")
    with torch.inference_mode(), tqdm(total=n - 1, desc=f"semantic sims (row-wise)") as pbar:
        for i in range(n - 1):
            cnt = n - i - 1
            if cnt <= 0:
                pbar.update(1); continue

            vi = emb[i].unsqueeze(0)                # (1, d)
            Vj = emb[i + 1 : n]                      # (cnt, d)

            # cosine since embeddings are normalized: sim = vi @ Vj^T ∈ [-1,1]
            sims = torch.matmul(Vj, vi.T).squeeze(1)  # (cnt,)
            sims = (sims + 1.0) * 0.5                 # map to [0,1]

            # move to CPU numpy and cast to on-disk dtype
            sims_np = sims.detach().to("cpu").numpy().astype(DTYPE, copy=False)
            feat_mm[cursor : cursor + cnt, col] = sims_np

            cursor += cnt
            pbar.update(1)

    feat_mm.flush()
    print("✅ Done. Semantic similarity written to", os.path.join(data_dir, FEAT_FILE))

if __name__ == "__main__":
    main()
