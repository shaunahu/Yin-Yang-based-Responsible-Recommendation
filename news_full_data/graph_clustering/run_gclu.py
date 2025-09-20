#!/usr/bin/env python3
# run_gclu_from_graph.py — cluster from NetworkX graph_with_edges.pkl, save labels + KxN cluster matrix + meta + manifest

import os
import json
import time
import heapq
import pickle
import numpy as np
from collections import defaultdict

# ---------------- paths (CONSISTENT) ----------------
HERE       = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR  = os.path.abspath(os.path.join(HERE, "..", "newsGraph"))
OUT_DIR    = os.path.abspath(os.path.join(HERE, "..", "saved_clusters"))

GRAPH_PKL  = os.path.join(GRAPH_DIR, "graph_with_edges.pkl")
# ----------------------------------------------------

# ---------------- clustering config -----------------
FEATURE_WEIGHTS = {
    "semantic_similarity": 0.4,
    "topic_similarity":     0.3,
    "sentiment_similarity": 0.2,
    "cooccurrence":         0.1,   # 如果图里用的键是 'frequent'，可改为 'frequent'
}
FEATURE_ALIASES = {
    "cooccurrence": ["cooccurrence", "frequent", "impression_cooccurrence_prob"],
    "semantic_similarity": ["semantic_similarity", "semantic_similarity_ta"],
    "topic_similarity": ["topic_similarity"],
    "sentiment_similarity": ["sentiment_similarity"],
}

TOPK          = 5     # symmetric top-k per node; None = 不裁剪
NUM_CLUSTERS  = 5
REPEATS       = 5
SEED          = 123
# ----------------------------------------------------


def load_graph(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph not found: {path}")
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G


def get_attr_with_alias(data: dict, name: str):
    if name in data:
        return data[name]
    for alt in FEATURE_ALIASES.get(name, []):
        if alt in data:
            return data[alt]
    return None


def edge_weight_from_attrs(data: dict, weights: dict) -> float:
    w = 0.0
    for name, alpha in weights.items():
        val = get_attr_with_alias(data, name)
        if val is None:
            continue
        try:
            w += float(alpha) * float(val)
        except Exception:
            continue
    return w


def symmetric_topk(edges_uv_w_labeled, k: int):
    """edges_uv_w_labeled: list of (node_label_u, node_label_v, weight)"""
    adj = defaultdict(list)
    for idx, (u, v, w) in enumerate(edges_uv_w_labeled):
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
    return [edges_uv_w_labeled[i] for i in sorted(keep)]


def normalize_labels(labels: np.ndarray):
    uniq = np.unique(labels)
    remap = {old: new for new, old in enumerate(uniq)}
    new_labels = np.array([remap[c] for c in labels], dtype=np.int64)
    return new_labels, remap


def save_cluster_matrix(labels: np.ndarray, nodes: list, out_dir: str, tag: str):
    """
    Save a KxN membership matrix (bool).
    Prefer torch .pt if available; otherwise save numpy .npy.
    """
    K = int(labels.max()) + 1
    N = labels.shape[0]
    try:
        import torch
        M = torch.zeros((K, N), dtype=torch.bool)
        lab_t = torch.from_numpy(labels)
        for k in range(K):
            M[k] = (lab_t == k)
        out_pt = os.path.join(out_dir, f"cluster_matrix_{tag}.pt")
        torch.save(M, out_pt)
        return {"format": "pt", "path": out_pt, "shape": [K, N], "dtype": "bool"}
    except Exception as e:
        M = np.zeros((K, N), dtype=bool)
        for i, cid in enumerate(labels.tolist()):
            M[int(cid), i] = True
        out_npy = os.path.join(out_dir, f"cluster_matrix_{tag}.npy")
        np.save(out_npy, M)
        return {"format": "npy", "path": out_npy, "shape": [K, N], "dtype": "bool", "note": str(e)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

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
        for name in FEATURE_WEIGHTS.keys():
            if get_attr_with_alias(data, name) is not None:
                used_features.add(name)

    print(f"[INFO] Edges with positive combined weight: {len(edges):,}")
    if not edges:
        raise RuntimeError("No edges with positive combined weight. Check feature names / weights.")

    # ---- Symmetric TOP-K ----
    if TOPK is not None:
        edges_labeled = [(nodes[u], nodes[v], w) for (u, v, w) in edges]
        edges_kept    = symmetric_topk(edges_labeled, k=TOPK)
        edges = [[node_index[u], node_index[v], float(w)] for (u, v, w) in edges_kept]
        print(f"[INFO] After symmetric TOPK={TOPK}: kept={len(edges):,}")

    # ---- Compress to active nodes for clustering ----
    src = np.array([e[0] for e in edges], dtype=np.int64)
    dst = np.array([e[1] for e in edges], dtype=np.int64)
    weights = np.array([e[2] for e in edges], dtype=np.float32)

    active_nodes = np.unique(np.concatenate([src, dst]))
    M = active_nodes.size
    old2new = {int(old): i for i, old in enumerate(active_nodes)}
    new2old = np.array(active_nodes, dtype=np.int64)

    src_c = np.vectorize(old2new.__getitem__)(src)
    dst_c = np.vectorize(old2new.__getitem__)(dst)

    print(f"[INFO] Nodes total={len(nodes):,}, active={M:,}, isolates={len(nodes)-M:,}")

    # ---- GCLU clustering ----
    try:
        from gclu import gclu
    except Exception:
        print("[ERROR] Could not import gclu. Install: https://github.com/uef-machine-learning/gclu")
        raise

    gclu_edges = [[int(u), int(v), float(w)] for u, v, w in zip(src_c, dst_c, weights)]
    print(f"[INFO] Running GCLU: edges={len(gclu_edges):,}, K={NUM_CLUSTERS}, repeats={REPEATS}, seed={SEED}")
    t1 = time.time()
    labels_active = gclu(gclu_edges,
                         graph_type="similarity",
                         num_clusters=NUM_CLUSTERS,
                         repeats=REPEATS,
                         scale="no",
                         seed=SEED,
                         costf="inv")
    t2 = time.time()
    print(f"[TIMING] GCLU took {t2 - t1:.2f}s")
    labels_active = np.asarray(labels_active, dtype=np.int32)
    if labels_active.shape[0] != M:
        raise RuntimeError(f"GCLU returned {len(labels_active)} labels, but active nodes = {M}")

    # ---- Expand to full N and assign isolates to largest cluster ----
    labels_full = np.full(len(nodes), -1, dtype=np.int32)
    labels_full[new2old] = labels_active
    if np.any(labels_full == -1):
        uniq, counts = np.unique(labels_active, return_counts=True)
        largest_cid = int(uniq[np.argmax(counts)])
        num_isolates = int(np.sum(labels_full == -1))
        labels_full[labels_full == -1] = largest_cid
        print(f"[INFO] Assigned {num_isolates} isolated nodes to largest cluster {largest_cid}.")

    # ---- Normalize IDs to 0..K-1 for stable outputs ----
    labels_norm, remap = normalize_labels(labels_full)
    K = int(labels_norm.max()) + 1
    tag = f"K{K}_topk{TOPK or 'all'}"

    # ---- Save labels ----
    labels_path = os.path.join(OUT_DIR, f"labels_{tag}.npy")
    np.save(labels_path, labels_norm)
    print(f"[INFO] Saved labels -> {labels_path}")

    # ---- Save cluster matrix ----
    matrix_info = save_cluster_matrix(labels_norm, nodes, OUT_DIR, tag)
    print(f"[INFO] Saved matrix -> {matrix_info['path']}  shape={matrix_info['shape']}")

    # ---- Build cluster summaries & manifest (✅ 新增：保存 manifest) ----
    cluster_nodes = defaultdict(list)
    item_to_cluster = {}
    for idx, cid in enumerate(labels_norm.tolist()):
        item_id = str(nodes[idx])
        cluster_nodes[int(cid)].append(item_id)
        item_to_cluster[item_id] = int(cid)

    cluster_edges = defaultdict(int)
    for u, v, _ in edges:
        cu, cv = labels_norm[u], labels_norm[v]
        if cu == cv:
            cluster_edges[int(cu)] += 1

    cluster_info = {}
    for cid, nodelist in cluster_nodes.items():
        cluster_info[int(cid)] = {
            "num_nodes": len(nodelist),
            "num_intra_edges": int(cluster_edges.get(int(cid), 0)),
        }
        print(f"[SUMMARY] Cluster {cid}: {len(nodelist)} nodes, {cluster_edges.get(cid,0)} intra-edges")

    # ---- Save run meta ----
    run_meta = {
        "tag": tag,
        "graph_file": os.path.basename(GRAPH_PKL),
        "num_nodes": int(len(nodes)),
        "num_edges_total": int(G.number_of_edges()),
        "num_edges_used": int(len(edges)),
        "feature_weights": FEATURE_WEIGHTS,
        "features_used": sorted(list(FEATURE_WEIGHTS.keys())),
        "aliases": FEATURE_ALIASES,
        "topk": int(TOPK) if TOPK is not None else None,
        "num_clusters": int(K),
        "repeats": int(REPEATS),
        "seed": int(SEED),
        "runtime_sec": round(time.time() - t0, 2),
        "labels_path": labels_path,
        "matrix": matrix_info,
        "clusters": cluster_info,
        "id_remap_note": "labels were normalized to 0..K-1",
    }
    meta_path = os.path.join(OUT_DIR, f"clustering_run_meta_{tag}.json")
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"[INFO] Saved run meta -> {meta_path}")

    # ---- Save cluster_matrix manifest (item_id ↔ cluster_id 互查) ----
    cluster_to_items = {str(k): cluster_nodes.get(k, []) for k in range(K)}
    manifest = {
        "tag": tag,
        "num_clusters": K,
        "num_nodes": len(nodes),
        "labels_path": os.path.abspath(labels_path),
        "matrix": matrix_info,
        "cluster_sizes": {str(k): len(cluster_to_items[str(k)]) for k in range(K)},
        "cluster_to_items": cluster_to_items,
        "item_to_cluster": item_to_cluster,
        "id_remap_note": "labels were normalized to 0..K-1"
    }
    manifest_path = os.path.join(OUT_DIR, f"cluster_matrix_manifest_{tag}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Saved cluster matrix manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
