#!/usr/bin/env python3
# run_gclu_from_graph.py — cluster from NetworkX graph_with_edges.pkl, save labels, matrix, meta, manifest, clustered graph, cluster medoids + distance-to-center

import os
import json
import time
import heapq
import pickle
import random
import numpy as np
from collections import defaultdict
import networkx as nx

# -------- Paths --------
HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR = os.path.abspath(os.path.join(HERE, "..", "newsGraph"))
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "saved_clusters"))
GRAPH_PKL = os.path.join(GRAPH_DIR, "graph_with_edges.pkl")

# -------- Clustering config --------
FEATURE_WEIGHTS = {
    "semantic_similarity": 0.4,
    "topic_similarity": 0.3,
    "sentiment_similarity": 0.2,
    "cooccurrence": 0.1,
}
FEATURE_ALIASES = {
    "cooccurrence": ["cooccurrence", "frequent", "impression_cooccurrence_prob"],
    "semantic_similarity": ["semantic_similarity", "semantic_similarity_ta"],
    "topic_similarity": ["topic_similarity"],
    "sentiment_similarity": ["sentiment_similarity"],
}
TOPK = 5          # Symmetric top-k pruning
NUM_CLUSTERS = 5
REPEATS = 5
SEED = 123

# -------- Medoid config --------
MEDOID_EXACT_THRESHOLD = 4000   # Exact BFS if cluster size <= threshold
MEDOID_SAMPLES_LARGE = 256      # Otherwise sample candidates
RNG = random.Random(SEED)

# -------- Utility --------
def load_graph(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

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
        if val is not None:
            try:
                w += float(alpha) * float(val)
            except:
                pass
    return w

def symmetric_topk(edge_list, k: int):
    # Keep top-k edges per node (symmetric)
    adj = defaultdict(list)
    for idx, (u, v, w) in enumerate(edge_list):
        adj[u].append((v, idx, w))
        adj[v].append((u, idx, w))
    keep = set()
    for u, neighs in adj.items():
        if len(neighs) <= k:
            for _, idx, _ in neighs:
                keep.add(idx)
        else:
            for _, idx, _ in heapq.nlargest(k, neighs, key=lambda x: x[2]):
                keep.add(idx)
    return [edge_list[i] for i in sorted(keep)]

def normalize_labels(labels: np.ndarray):
    uniq = np.unique(labels)
    remap = {old: new for new, old in enumerate(uniq)}
    new_labels = np.array([remap[c] for c in labels], dtype=np.int64)
    return new_labels, remap

def save_cluster_matrix(labels: np.ndarray, nodes, out_dir, tag):
    # Save cluster membership matrix KxN, prefer torch if available
    K = int(labels.max()) + 1
    N = len(nodes)
    try:
        import torch
        M = torch.zeros((K, N), dtype=torch.bool)
        lab = torch.from_numpy(labels)
        for k in range(K):
            M[k] = (lab == k)
        path = os.path.join(out_dir, f"cluster_matrix_{tag}.pt")
        torch.save(M, path)
        return {"format": "pt", "path": path, "shape": [K, N], "dtype": "bool"}
    except:
        M = np.zeros((K, N), bool)
        for i, c in enumerate(labels):
            M[int(c), i] = True
        path = os.path.join(out_dir, f"cluster_matrix_{tag}.npy")
        np.save(path, M)
        return {"format": "npy", "path": path, "shape": [K, N], "dtype": "bool"}

# -------- Medoid & distance helpers --------
def _bfs_dist_sum(G, cluster_set, source):
    # BFS from source, sum distances only for nodes in cluster_set
    dist_map = {}
    lengths = nx.single_source_shortest_path_length(G, source)
    for n, d in lengths.items():
        if n in cluster_set:
            dist_map[n] = d
    total = sum(dist_map.values())
    return total, dist_map

def find_cluster_medoids_unweighted(G, cluster_nodes):
    medoids = {}
    medoid_dists = {}
    for cid, nlist in cluster_nodes.items():
        cluster_set = set(nlist)
        size = len(cluster_set)
        print(f"[MEDOID] Cluster {cid}: size={size}")
        # Select candidates
        candidates = list(nlist)
        if size > MEDOID_EXACT_THRESHOLD:
            RNG.shuffle(candidates)
            candidates = candidates[:MEDOID_SAMPLES_LARGE]
            print(f"[MEDOID] Sampling {len(candidates)} candidates")
        best_node = None
        best_sum = float("inf")
        best_map = None
        for i, node in enumerate(candidates, 1):
            total, d_map = _bfs_dist_sum(G, cluster_set, node)
            if total < best_sum:
                best_sum = total
                best_node = node
                best_map = d_map
        medoids[cid] = best_node
        medoid_dists[cid] = best_map if best_map else {}
        print(f"[CENTER] Cluster {cid}: center={best_node}, sumDist={best_sum}")
    return medoids, medoid_dists

def build_distance_vector(nodes_order, medoids, medoid_dists, unreachable=-1):
    dist_vec = np.full(len(nodes_order), unreachable, dtype=np.int64)
    idx_map = {str(n): i for i, n in enumerate(nodes_order)}
    for cid, center in medoids.items():
        for node, d in medoid_dists[cid].items():
            idx = idx_map.get(str(node))
            if idx is not None:
                dist_vec[idx] = d
        if center is not None:
            idxc = idx_map.get(str(center))
            if idxc is not None:
                dist_vec[idxc] = 0
    return dist_vec

def try_save_tensor(dist_vec, out_base):
    try:
        import torch
        t = torch.from_numpy(dist_vec)
        path = f"{out_base}.pt"
        torch.save(t, path)
        print(f"[SAVE] distances -> {path}")
        return {"path": path, "format": "pt", "shape": list(t.shape)}
    except:
        npy = f"{out_base}.npy"
        np.save(npy, dist_vec)
        print(f"[SAVE] distances -> {npy}")
        return {"path": npy, "format": "npy", "shape": list(dist_vec.shape)}

# -------- Main --------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(SEED)

    # Load graph
    t0 = time.time()
    G = load_graph(GRAPH_PKL)
    nodes = list(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    print(f"[INFO] Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, time={time.time()-t0:.2f}s")

    # Build weighted edges
    edges = []
    for u, v, data in G.edges(data=True):
        w = edge_weight_from_attrs(data, FEATURE_WEIGHTS)
        if w > 0:
            edges.append([node_index[u], node_index[v], w])
    print(f"[INFO] Positive edges: {len(edges)}")

    if TOPK is not None:
        edges_l = [(nodes[u], nodes[v], w) for u, v, w in edges]
        edges_l = symmetric_topk(edges_l, TOPK)
        edges = [[node_index[u], node_index[v], w] for u, v, w in edges_l]
        print(f"[INFO] After TOPK={TOPK}: {len(edges)} edges")

    # Compress active nodes
    src = np.array([e[0] for e in edges])
    dst = np.array([e[1] for e in edges])
    active = np.unique(np.concatenate([src, dst]))
    old2new = {old: i for i, old in enumerate(active)}
    src_c = np.vectorize(old2new.__getitem__)(src)
    dst_c = np.vectorize(old2new.__getitem__)(dst)
    print(f"[INFO] Active nodes: {len(active)} / total {len(nodes)}")

    # Run GCLU
    try:
        from gclu import gclu
    except:
        raise ImportError("Install gclu: https://github.com/uef-machine-learning/gclu")

    edges_gclu = [[int(u), int(v), float(w)] for u, v, w in zip(src_c, dst_c, [e[2] for e in edges])]
    print(f"[INFO] Running GCLU...")
    t1 = time.time()
    labels_active = gclu(edges_gclu, graph_type="similarity", num_clusters=NUM_CLUSTERS,
                         repeats=REPEATS, scale="no", seed=SEED, costf="inv")
    t2 = time.time()
    print(f"[INFO] GCLU done in {t2 - t1:.2f}s")
    labels_active = np.array(labels_active, dtype=np.int32)

    # Expand labels for all nodes
    labels_full = np.full(len(nodes), -1, np.int32)
    for i, a in enumerate(active):
        labels_full[a] = labels_active[i]
    if (labels_full == -1).any():
        unique, counts = np.unique(labels_active, return_counts=True)
        largest = unique[np.argmax(counts)]
        labels_full[labels_full == -1] = largest

    # Normalize labels
    labels_norm, _ = normalize_labels(labels_full)
    tag = f"K{labels_norm.max()+1}_topk{TOPK or 'all'}"

    # Save labels & matrix
    labels_path = os.path.join(OUT_DIR, f"labels_{tag}.npy")
    np.save(labels_path, labels_norm)
    matrix_info = save_cluster_matrix(labels_norm, nodes, OUT_DIR, tag)

    # Build clusters
    cluster_nodes = defaultdict(list)
    for i, c in enumerate(labels_norm):
        cluster_nodes[int(c)].append(str(nodes[i]))

    # Save manifest
    manifest = {
        "tag": tag,
        "num_clusters": int(labels_norm.max()+1),
        "num_nodes": len(nodes),
        "labels_path": os.path.abspath(labels_path),
        "matrix": matrix_info,
        "cluster_sizes": {str(k): len(v) for k, v in cluster_nodes.items()}
    }
    manifest_path = os.path.join(OUT_DIR, f"cluster_matrix_manifest_{tag}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Saved manifest -> {manifest_path}")

    # Save graph with cluster labels
    for i, node in enumerate(nodes):
        G.nodes[node]["cluster"] = int(labels_norm[i])
    graph_clustered_path = os.path.join(OUT_DIR, f"graph_with_clusters_{tag}.pkl")
    with open(graph_clustered_path, "wb") as f:
        pickle.dump(G, f)
    print(f"[INFO] Saved clustered graph -> {graph_clustered_path}")

    # --- Medoid + distance-to-center ---
    # Save node order
    nodes_order_path = os.path.join(OUT_DIR, f"nodes_order_{tag}.txt")
    with open(nodes_order_path, "w") as f:
        for n in nodes:
            f.write(str(n) + "\n")

    medoids, medoid_dists = find_cluster_medoids_unweighted(G, cluster_nodes)

    # Save medoids
    centers_path = os.path.join(OUT_DIR, f"cluster_centers_{tag}.json")
    with open(centers_path, "w") as f:
        json.dump({cid: n for cid, n in medoids.items()}, f, indent=2)

    # Build global distance vector
    dist_vec = build_distance_vector(nodes, medoids, medoid_dists)
    dist_base = os.path.join(OUT_DIR, f"cluster_distances_{tag}")
    dist_info = try_save_tensor(dist_vec, dist_base)

    print("[DONE] Clustering + medoid calculation complete.")

if __name__ == "__main__":
    main()
