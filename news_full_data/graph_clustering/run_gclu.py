#!/usr/bin/env python3
# run_gclu_from_graph.py — cluster from NetworkX graph_with_edges.pkl and save cluster info

import os, json, time, heapq, pickle
import numpy as np
from collections import defaultdict

# ---------------- config ----------------
HERE       = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR  = os.path.abspath(os.path.join(HERE, "..", "newsGraph"))
OUT_DIR    = os.path.abspath(os.path.join(HERE, "..", "saved_clusters"))

GRAPH_PKL  = os.path.join(GRAPH_DIR, "graph_with_edges.pkl")  # input graph

FEATURE_WEIGHTS = {
    "semantic_similarity": 0.4,
    "topic_similarity":     0.3,
    "sentiment_similarity": 0.2,
    "cooccurrence":         0.1,   # or "frequent" if that’s the name in your graph
}

TOPK = 5
NUM_CLUSTERS = 5
REPEATS = 5
SEED = 123
# ----------------------------------------


def load_graph(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph not found: {path}")
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G


def edge_weight_from_attrs(data: dict, weights: dict) -> float:
    w = 0.0
    for name, alpha in weights.items():
        if name not in data:
            continue
        try:
            w += float(alpha) * float(data[name])
        except Exception:
            continue
    return w


def symmetric_topk(edges_uv_w, k: int):
    """Symmetric Top-K pruning: keep edge if in top-k of either endpoint."""
    adj = defaultdict(list)
    for idx, (u, v, w) in enumerate(edges_uv_w):
        adj[u].append((v, idx, w))
        adj[v].append((u, idx, w))

    keep = set()
    for u, neigh_list in adj.items():
        if len(neigh_list) <= k:
            for _, idx, _ in neigh_list:
                keep.add(idx)
        else:
            for _, idx, _ in heapq.nlargest(k, neigh_list, key=lambda t: t[2]):
                keep.add(idx)

    return [edges_uv_w[i] for i in sorted(keep)]


def main():
    # ---- Load graph ----
    t0 = time.time()
    G = load_graph(GRAPH_PKL)
    nodes = list(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    print(f"[INFO] Loaded graph: nodes={G.number_of_nodes():,}, edges={G.number_of_edges():,}  ({time.time()-t0:.2f}s)")

    # ---- Build weighted edges ----
    edges = []
    used_features = set()
    for u, v, data in G.edges(data=True):
        w = edge_weight_from_attrs(data, FEATURE_WEIGHTS)
        if w <= 0.0:
            continue
        edges.append([node_index[u], node_index[v], float(w)])
        for name in FEATURE_WEIGHTS:
            if name in data:
                used_features.add(name)

    print(f"[INFO] Edges with positive combined weight: {len(edges):,}")
    if not edges:
        raise RuntimeError("No edges with positive combined weight. Check feature names / weights.")

    # ---- Symmetric TOP-K ----
    if TOPK is not None:
        edges_label = [(nodes[u], nodes[v], w) for (u, v, w) in edges]
        edges_kept  = symmetric_topk(edges_label, k=TOPK)
        edges = [[node_index[u], node_index[v], float(w)] for (u, v, w) in edges_kept]
        print(f"[INFO] After symmetric TOPK={TOPK}: kept={len(edges):,}")

    # ---- Run GCLU ----
    try:
        from gclu import gclu
    except Exception:
        print("[ERROR] Could not import gclu. Install gclu from https://github.com/uef-machine-learning/gclu")
        raise

    print(f"[INFO] Running GCLU: K={NUM_CLUSTERS}, repeats={REPEATS}, seed={SEED}")
    t1 = time.time()
    labels = gclu(edges,
                  graph_type="similarity",
                  num_clusters=NUM_CLUSTERS,
                  repeats=REPEATS,
                  scale="no",
                  seed=SEED,
                  costf="inv")
    t2 = time.time()
    print(f"[TIMING] GCLU took {t2 - t1:.2f}s")

    labels = np.asarray(labels, dtype=np.int32)
    tag = f"K{NUM_CLUSTERS}_topk{TOPK or 'all'}"

    # ---- Save labels ----
    os.makedirs(OUT_DIR, exist_ok=True)
    labels_path = os.path.join(OUT_DIR, f"labels_{tag}.npy")
    np.save(labels_path, labels)
    print(f"[INFO] Saved labels -> {labels_path}")

    # ---- Compute cluster info ----
    cluster_nodes = defaultdict(list)
    for idx, cid in enumerate(labels):
        cluster_nodes[int(cid)].append(nodes[idx])

    cluster_edges = defaultdict(int)
    for u, v, _ in edges:
        cu, cv = labels[u], labels[v]
        if cu == cv:
            cluster_edges[int(cu)] += 1

    cluster_info = {}
    for cid, nodelist in cluster_nodes.items():
        cluster_info[cid] = {
            "num_nodes": len(nodelist),
            "num_intra_edges": cluster_edges.get(cid, 0),
        }
        print(f"[SUMMARY] Cluster {cid}: {len(nodelist)} nodes, {cluster_edges.get(cid,0)} intra-edges")

    # ---- Save run meta + cluster info ----
    run_meta = {
        "tag": tag,
        "graph_file": os.path.basename(GRAPH_PKL),
        "num_nodes": int(G.number_of_nodes()),
        "num_edges_total": int(G.number_of_edges()),
        "num_edges_used": int(len(edges)),
        "feature_weights": FEATURE_WEIGHTS,
        "features_used": sorted(list(used_features)),
        "topk": int(TOPK) if TOPK is not None else None,
        "num_clusters": int(NUM_CLUSTERS),
        "repeats": int(REPEATS),
        "seed": int(SEED),
        "runtime_sec": round(t2 - t0, 2),
        "clusters": cluster_info
    }
    meta_path = os.path.join(OUT_DIR, f"clustering_run_meta_{tag}.json")
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"[INFO] Saved run meta -> {meta_path}")


if __name__ == "__main__":
    main()
