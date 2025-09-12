import os
import json
import argparse
import numpy as np
from collections import defaultdict

HERE       = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR  = os.path.abspath(os.path.join(HERE, "..", "newsGraph"))
DATA_DIR   = os.path.abspath(os.path.join(HERE, "..", "data_news"))
OUT_DIR    = os.path.abspath(os.path.join(HERE, "..", "saved_clusters"))

# Node embedding cache (aligned to nodes.npy)
NODE_EMB_PATH = os.path.join(GRAPH_DIR, "node_embeddings_minilm.npy")
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH   = 512

# Which edge features to include (friendly names; resolved via features_meta.json)
DEFAULT_EDGE_FEATURES = [
    "semantic_similarity",          # will also try alias 'semantic_similarity_ta'
    "topic_similarity",
    "sentiment_similarity",
    "frequent",                     # will also try alias 'impression_cooccurrence_prob'
]

def resolve_feature_columns(meta, requested):
    """Map friendly feature names to actual columns in edge_feats.npy (with aliases)."""
    feats = meta.get("features", {})
    aliases = {
        "semantic_similarity": ["semantic_similarity", "semantic_similarity_ta"],
        "topic_similarity": ["topic_similarity"],
        "sentiment_similarity": ["sentiment_similarity"],
        "frequent": ["frequent", "impression_cooccurrence_prob"],
    }
    resolved = []
    for name in requested:
        for c in aliases.get(name, [name]):
            if c in feats:
                resolved.append((name, c, int(feats[c])))
                break
        else:
            print(f"[WARN] Feature '{name}' not found; skipping.")
    if not resolved:
        raise RuntimeError("No requested edge features found in features_meta.json")
    return resolved  # list of (friendly, actual_key, col_idx)

def get_or_build_node_embeddings(nodes):
    """Return (n,d) node embeddings aligned to nodes.npy. Build from items_filtered.csv if missing."""
    if os.path.exists(NODE_EMB_PATH):
        emb = np.load(NODE_EMB_PATH, allow_pickle=True)
        if isinstance(emb, np.ndarray) and emb.ndim == 2 and emb.shape[0] == len(nodes):
            print(f"[INFO] Loaded cached node embeddings: {emb.shape} -> {NODE_EMB_PATH}")
            return emb.astype(np.float32, copy=False)

        print(f"[WARN] Cached embeddings shape mismatch: found {emb.shape}, expected ({len(nodes)}, d); rebuilding.")

    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer

    items_csv = os.path.join(DATA_DIR, "items_filtered.csv")
    if not os.path.exists(items_csv):
        raise FileNotFoundError(f"Missing items_filtered.csv at {items_csv}")

    df = pd.read_csv(items_csv, dtype=str)
    id2row = {iid: idx for idx, iid in enumerate(df["item_id"].tolist())}
    texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()
    texts = [texts[id2row[iid]] for iid in nodes]  # align to nodes order

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Building node embeddings with {EMBED_MODEL} on {device}")
    model = SentenceTransformer(EMBED_MODEL, device=device)
    with torch.inference_mode():
        emb = model.encode(
            texts, batch_size=EMBED_BATCH, convert_to_tensor=False,
            device=device, normalize_embeddings=True
        ).astype(np.float32, copy=False)

    os.makedirs(os.path.dirname(NODE_EMB_PATH), exist_ok=True)
    np.save(NODE_EMB_PATH, emb)
    print(f"[INFO] Saved node embeddings -> {NODE_EMB_PATH}")
    return emb

def compute_node_centroids(labels, nodes):
    """Mean (unit-normalized) node embeddings per cluster."""
    emb = get_or_build_node_embeddings(nodes)  # (n,d)
    labels = np.asarray(labels, dtype=np.int32)
    clusters = defaultdict(list)
    for idx, cid in enumerate(labels):
        clusters[int(cid)].append(idx)

    centroids = {}
    for cid, idxs in clusters.items():
        sub = emb[idxs]
        c = sub.mean(axis=0)
        c /= (np.linalg.norm(c) + 1e-8)
        centroids[cid] = c.astype(np.float32)
    return centroids  # cid -> (d,)

def load_edges_uppertri_memmap(n_nodes):
    """Open upper-triangular edges as memmaps (raw, no .npy header); return (src, dst)."""
    m = n_nodes * (n_nodes - 1) // 2
    src = np.memmap(os.path.join(GRAPH_DIR, "edges_src_int32.npy"),
                    dtype=np.int32, mode="r", shape=(m,))
    dst = np.memmap(os.path.join(GRAPH_DIR, "edges_dst_int32.npy"),
                    dtype=np.int32, mode="r", shape=(m,))
    return src, dst  # memmaps

def stream_edge_feature_means_per_cluster(labels, edge_cols):
    """
    Stream through upper-tri edge order; accumulate per-cluster sums of the selected feature columns
    over **intra-cluster** edges. Return dict: cid -> (mean_vector, count).
    """
    nodes = np.load(os.path.join(GRAPH_DIR, "nodes.npy"), allow_pickle=True)
    n = len(nodes)
    src, dst = load_edges_uppertri_memmap(n)

    # memmap edge feature matrix
    meta_path = os.path.join(GRAPH_DIR, "features_meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    num_cols = int(meta["num_cols"])

    feat_mm = np.memmap(os.path.join(GRAPH_DIR, "edge_feats.npy"),
                        dtype=np.float16, mode="r", shape=(len(src), num_cols))

    labels = np.asarray(labels, dtype=np.int32)

    # accumulators
    k = len(edge_cols)
    sum_by_cluster = defaultdict(lambda: np.zeros(k, dtype=np.float64))
    cnt_by_cluster = defaultdict(int)

    # stream rows in upper-tri layout
    cursor = 0
    for i in range(n - 1):
        cnt = n - i - 1
        if cnt <= 0:
            continue
        row_start = cursor
        row_end   = cursor + cnt

        # endpoints for this row are (i, j=i+1..n-1)
        c_i = int(labels[i])
        js = dst[row_start:row_end]

        # Identify which edges in this row are intra-cluster
        # We’ll do it in numpy for speed:
        c_j = labels[js]  # vector of cluster ids for j
        mask = (c_j == c_i)
        if not np.any(mask):
            cursor = row_end
            continue

        # rows to grab in feat_mm
        row_idx = np.nonzero(mask)[0]
        if row_idx.size > 0:
            # gather selected feature columns; note feat_mm is (m, num_cols)
            # Do a single slice per column to stay memory-light.
            vals = []
            for _, _, col in edge_cols:
                col_slice = np.asarray(feat_mm[row_start:row_end, col], dtype=np.float32)
                # treat NaN as 0.0
                if np.isnan(col_slice).any():
                    col_slice = np.nan_to_num(col_slice, nan=0.0)
                vals.append(col_slice[row_idx])  # only intra-cluster edges
            # shape: (k, num_intra_edges)
            V = np.vstack(vals)
            sum_by_cluster[c_i] += V.sum(axis=1)
            cnt_by_cluster[c_i] += V.shape[1]

        cursor = row_end

    # compute means
    mean_by_cluster = {}
    for cid in sum_by_cluster:
        c = cnt_by_cluster[cid]
        if c > 0:
            mean_by_cluster[cid] = (sum_by_cluster[cid] / c).astype(np.float32)
        else:
            mean_by_cluster[cid] = np.zeros(k, dtype=np.float32)

    return mean_by_cluster, cnt_by_cluster

def save_hybrid_embeddings(labels_path, edge_feature_names=None):
    """
    Create a hybrid embedding per cluster:
      [ node_centroid (d dims) || mean(edge_features_selected) (k dims) ]
    Saves one .npy per cluster and a manifest JSON with sizes/column names.
    """
    if edge_feature_names is None:
        edge_feature_names = DEFAULT_EDGE_FEATURES

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    # derive tag from labels filename
    tag = os.path.splitext(os.path.basename(labels_path))[0].replace("labels_", "")

    labels = np.load(labels_path, allow_pickle=True).astype(np.int32)
    nodes  = np.load(os.path.join(GRAPH_DIR, "nodes.npy"), allow_pickle=True)
    if labels.shape[0] != nodes.shape[0]:
        raise ValueError(f"Labels length {labels.shape[0]} != nodes length {nodes.shape[0]}")

    # 1) node centroids (d dims)
    node_centroids = compute_node_centroids(labels, nodes)
    d = next(iter(node_centroids.values())).shape[0]

    # 2) resolve edge feature columns & stream per-cluster means (k dims)
    with open(os.path.join(GRAPH_DIR, "features_meta.json"), "r") as f:
        meta = json.load(f)
    edge_cols = resolve_feature_columns(meta, edge_feature_names)  # list of (friendly, key, col_idx)
    mean_edge_feats, cnt_edges = stream_edge_feature_means_per_cluster(labels, edge_cols)
    k = len(edge_cols)

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {
        "tag": tag,
        "node_embedding_dim": int(d),
        "edge_feature_names": [name for (name, _, _) in edge_cols],
        "edge_feature_cols": {name: int(col) for (name, _, col) in edge_cols},
        "clusters": {}
    }

    # 3) build hybrid vectors and save
    for cid in sorted(node_centroids.keys()):
        node_vec = node_centroids[cid]                       # (d,)
        feat_vec = mean_edge_feats.get(cid, np.zeros(k, np.float32))  # (k,)

        hybrid = np.concatenate([node_vec, feat_vec], axis=0).astype(np.float32)
        path = os.path.join(OUT_DIR, f"cluster_{cid}_hybrid_{tag}.npy")
        np.save(path, hybrid)

        manifest["clusters"][str(cid)] = {
            "size_nodes": int((labels == cid).sum()),
            "size_intra_edges": int(cnt_edges.get(cid, 0)),
            "hybrid_path": path,
            "hybrid_dim": int(hybrid.shape[0]),
        }
        print(f"[INFO] Saved cluster {cid} hybrid embedding (dim={hybrid.shape[0]}) -> {path}")

    # 4) write manifest
    mf = os.path.join(OUT_DIR, f"cluster_hybrid_manifest_{tag}.json")
    with open(mf, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Wrote hybrid manifest -> {mf}")

def parse_args():
    ap = argparse.ArgumentParser(description="Compute hybrid cluster embeddings (nodes + edge features).")
    ap.add_argument("--labels", required=True,
                    help="Path to labels_*.npy produced by clustering (e.g., labels_K5_topk5.npy)")
    ap.add_argument("--features", nargs="*", default=None,
                    help="Edge feature names to include (defaults to semantic_similarity, topic_similarity, sentiment_similarity, frequent).")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    save_hybrid_embeddings(labels_path=args.labels, edge_feature_names=args.features)
