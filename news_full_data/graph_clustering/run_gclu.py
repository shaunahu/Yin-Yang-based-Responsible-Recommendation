# run_gclu.py  — minimal: cluster only, save labels + meta
import os, json, time, numpy as np
from collections import defaultdict
import heapq

# ---------------- config ----------------
HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR = os.path.abspath(os.path.join(HERE, "..", "newsGraph"))
SAVED_DIR = os.path.abspath(os.path.join(HERE, "..", "saved_clusters"))

# Choose/weight edge feature columns (resolved via features_meta.json)
FEATURE_WEIGHTS = {
    "semantic_similarity": 0.4,        # alias: semantic_similarity
    "topic_similarity":     0.3,
    "sentiment_similarity": 0.2,
    "frequent":             0.1        # alias: impression_cooccurrence_prob
}

TOPK = 5              # keep top-k strongest edges per node (symmetric); set None to keep all
NUM_CLUSTERS = 5
REPEATS = 5
SEED = 123
# ----------------------------------------

def resolve_feature_columns(meta, requested_weights):
    """Map friendly feature names -> (col_index, weight) using features_meta.json (with aliases)."""
    feats = meta.get("features", {})
    aliases = {
        "semantic_similarity": ["semantic_similarity", "semantic_similarity"],
        "topic_similarity": ["topic_similarity"],
        "sentiment_similarity": ["sentiment_similarity"],
        "frequent": ["frequent", "impression_cooccurrence_prob"],
    }
    resolved = {}
    for name, w in requested_weights.items():
        cols = aliases.get(name, [name])
        found = None
        for c in cols:
            if c in feats:
                found = feats[c]
                if c != name:
                    print(f"[INFO] Using alias '{c}' for requested feature '{name}'.")
                break
        if found is None:
            print(f"[WARN] Feature '{name}' not found; skipping.")
            continue
        resolved[name] = (int(found), float(w))
    if not resolved:
        raise RuntimeError("No requested features found in features_meta.json")
    return resolved


def load_edges_memmap(graph_dir: str, n_nodes: int):
    """Reopen raw memmap edge endpoints (upper-triangular layout)."""
    m = n_nodes * (n_nodes - 1) // 2
    src = np.memmap(os.path.join(graph_dir, "edges_src_int32.npy"),
                    dtype=np.int32, mode="r", shape=(m,))
    dst = np.memmap(os.path.join(graph_dir, "edges_dst_int32.npy"),
                    dtype=np.int32, mode="r", shape=(m,))
    # return dense views (keeps downstream simple; still zero-copy when possible)
    return np.array(src, copy=False), np.array(dst, copy=False)


def combine_weights_row_slice(feat_mm, mapping, row_start: int, cnt: int) -> np.ndarray:
    """Weighted-sum similarity for slice [row_start:row_start+cnt)."""
    out = np.zeros(cnt, dtype=np.float32)
    for _, (col, alpha) in mapping.items():
        col_view = np.asarray(feat_mm[row_start:row_start+cnt, col], dtype=np.float32)
        if np.isnan(col_view).any():
            col_view = np.nan_to_num(col_view, nan=0.0)
        out += alpha * col_view
    return out


def prune_topk_streaming_uppertri(src, dst, feat_mm, mapping, n_nodes: int, k: int):
    """
    Symmetric TOP-K with tiny memory, assuming edges are in upper-triangular order:
      for i in [0..n-2]: edges (i, j) for j=i+1..n-1 laid out contiguously.
    Keeps an edge if it's in top-k of either endpoint.
    Returns filtered (src_out, dst_out, weights_out).
    If k is None, returns all edges + weights.
    """
    if k is None:
        m = len(src)
        W = np.empty(m, dtype=np.float32)
        cursor = 0
        for i in range(n_nodes - 1):
            cnt = n_nodes - i - 1
            if cnt <= 0: continue
            W[cursor:cursor+cnt] = combine_weights_row_slice(feat_mm, mapping, cursor, cnt)
            cursor += cnt
        return src, dst, W

    kept_a = set()
    heaps_b = [ [] for _ in range(n_nodes) ]  # per-destination min-heaps (w, idx)
    cursor = 0

    # Pass A: top-k per source i; also feed destination heaps for Pass B
    t0 = time.time()
    for i in range(n_nodes - 1):
        cnt = n_nodes - i - 1
        if cnt <= 0: continue
        row_start = cursor
        row_end   = cursor + cnt

        w_row = combine_weights_row_slice(feat_mm, mapping, row_start, cnt)

        # top-k for source i
        if cnt <= k:
            kept_a.update(range(row_start, row_end))
        else:
            top_idx_local = np.argpartition(w_row, -k)[-k:]
            kept_a.update(row_start + top_idx_local)

        # feed destination heaps
        js = dst[row_start:row_end]
        for off in range(cnt):
            idx = row_start + off
            j   = int(js[off])
            wt  = float(w_row[off])
            hb  = heaps_b[j]
            if len(hb) < k:
                heapq.heappush(hb, (wt, idx))
            elif wt > hb[0][0]:
                heapq.heapreplace(hb, (wt, idx))

        cursor = row_end
    t1 = time.time()
    print(f"[TIMING] Pass-A (per-source top-{k} + feed B): {t1 - t0:.2f}s")

    # Collect pass-B keep set
    kept_b = set()
    for j in range(n_nodes):
        if heaps_b[j]:
            kept_b.update(idx for (_, idx) in heaps_b[j])

    keep_idx = np.fromiter(kept_a.union(kept_b), dtype=np.int64)
    keep_idx.sort()

    # Rebuild weights only for kept edges (row-batched)
    W_kept = np.empty(len(keep_idx), dtype=np.float32)
    cursor = 0
    p = 0
    t2 = time.time()
    for i in range(n_nodes - 1):
        cnt = n_nodes - i - 1
        if cnt <= 0: continue
        row_start = cursor
        row_end   = cursor + cnt

        if p < len(keep_idx) and keep_idx[p] < row_end:
            w_row = combine_weights_row_slice(feat_mm, mapping, row_start, cnt)
            while p < len(keep_idx) and row_start <= keep_idx[p] < row_end:
                W_kept[p] = w_row[keep_idx[p] - row_start]
                p += 1

        cursor = row_end
    t3 = time.time()
    print(f"[TIMING] Weights rebuild for kept edges: {t3 - t2:.2f}s")

    return src[keep_idx], dst[keep_idx], W_kept


def main():
    # Load nodes
    nodes = np.load(os.path.join(GRAPH_DIR, "nodes.npy"), allow_pickle=True)
    n = len(nodes)

    # Load edges (memmap format)
    t = time.time()
    src, dst = load_edges_memmap(GRAPH_DIR, n)
    print(f"[INFO] Nodes={n:,}, raw edges={len(src):,}  (loaded in {time.time()-t:.2f}s)")

    # Load feature meta + matrix
    with open(os.path.join(GRAPH_DIR, "features_meta.json"), "r") as f:
        meta = json.load(f)
    mapping = resolve_feature_columns(meta, FEATURE_WEIGHTS)

    feat_mm = np.memmap(
        os.path.join(GRAPH_DIR, "edge_feats.npy"),
        dtype=np.float16, mode="r",
        shape=(len(src), int(meta["num_cols"]))
    )
    print(f"[INFO] Using features: " + ", ".join([f"{k}:{v[0]}" for k,v in mapping.items()]))

    # Streaming symmetric TOP-K pruning
    t = time.time()
    src, dst, weights = prune_topk_streaming_uppertri(
        src=src, dst=dst, feat_mm=feat_mm, mapping=mapping, n_nodes=n, k=TOPK
    )
    print(f"[INFO] Kept edges={len(src):,} after symmetric TOPK={TOPK}  ({time.time()-t:.2f}s)")

    # Run GCLU
    try:
        from gclu import gclu
    except Exception:
        print("[ERROR] Could not import gclu. Install/clone the repo so `from gclu import gclu` works.")
        print("        Repo: https://github.com/uef-machine-learning/gclu")
        raise

    edges = [[int(u), int(v), float(w)] for u, v, w in zip(src, dst, weights)]
    print(f"[INFO] Running GCLU on {len(edges):,} edges, K={NUM_CLUSTERS}, repeats={REPEATS}, seed={SEED}")
    t = time.time()
    labels = gclu(edges,
                  graph_type="similarity",
                  num_clusters=NUM_CLUSTERS,
                  repeats=REPEATS,
                  scale="no",
                  seed=SEED,
                  costf="inv")
    print(f"[TIMING] GCLU: {time.time() - t:.2f}s")

    labels = np.asarray(labels, dtype=np.int32)
    tag = f"K{NUM_CLUSTERS}_topk{TOPK or 'all'}"

    # Save labels (this is what cluster_embeddings_hybrid.py needs)
    os.makedirs(SAVED_DIR, exist_ok=True)
    labels_path = os.path.join(HERE, f"labels_{tag}.npy")
    np.save(labels_path, labels)
    print(f"[INFO] Saved labels -> {labels_path}")

    # Also save a tiny run meta (nice for reproducibility)
    run_meta = {
        "tag": tag,
        "num_nodes": int(n),
        "num_edges_kept": int(len(src)),
        "num_clusters": int(NUM_CLUSTERS),
        "repeats": int(REPEATS),
        "seed": int(SEED),
        "topk": int(TOPK) if TOPK is not None else None,
        "feature_weights": FEATURE_WEIGHTS,
        "feature_cols": {k: int(col) for k, (col, _) in mapping.items()},
    }
    meta_path = os.path.join(HERE, f"clustering_run_meta_{tag}.json")
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"[INFO] Saved run meta -> {meta_path}")

    # Quick sizes
    uniq, counts = np.unique(labels, return_counts=True)
    print("[INFO] Cluster sizes:", dict(zip(uniq.tolist(), counts.tolist())))

if __name__ == "__main__":
    main()
