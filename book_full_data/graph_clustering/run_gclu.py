#!/usr/bin/env python3
# run_gclu_from_graph.py
# Cluster from NetworkX graph_with_edges.pkl using GCLU,
# but EXCLUDE isolated/inactive nodes completely from ALL downstream artifacts.

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
GRAPH_DIR = os.path.abspath(os.path.join(HERE, "..", "booksGraph"))
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "saved_clusters"))
GRAPH_PKL = os.path.join(GRAPH_DIR, "graph_with_edges.pkl")

# -------- Clustering config --------
FEATURE_WEIGHTS = {
    "semantic_similarity": 0.25,
    "topic_similarity": 0.25,
    "sentiment_similarity": 0.25,
    "cooccurrence": 0.25,
}
FEATURE_ALIASES = {
    "cooccurrence": ["cooccurrence", "frequent", "impression_cooccurrence_prob"],
    "semantic_similarity": ["semantic_similarity", "semantic_similarity_ta"],
    "topic_similarity": ["topic_similarity"],
    "sentiment_similarity": ["sentiment_similarity"],
}
TOPK = 15          # Symmetric top-k pruning
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
    """
    Keep top-k edges per node (symmetric).
    edge_list: list[(u_node, v_node, weight)]
    """
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
    """
    cluster_nodes: dict[int] -> list[node_id]  (node_id should match G's node IDs)
    """
    medoids = {}
    medoid_dists = {}
    for cid, nlist in cluster_nodes.items():
        cluster_set = set(nlist)
        size = len(cluster_set)
        print(f"[MEDOID] Cluster {cid}: size={size}")

        candidates = list(nlist)
        if size > MEDOID_EXACT_THRESHOLD:
            RNG.shuffle(candidates)
            candidates = candidates[:MEDOID_SAMPLES_LARGE]
            print(f"[MEDOID] Sampling {len(candidates)} candidates")

        best_node = None
        best_sum = float("inf")
        best_map = None
        for node in candidates:
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
                dist_vec[idx] = int(d)
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
    G_full = load_graph(GRAPH_PKL)
    print(f"[INFO] Loaded graph: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges, time={time.time()-t0:.2f}s")

    # Optional: remove degree-0 isolates in the ORIGINAL graph (cheap, safe)
    isolates_full = list(nx.isolates(G_full))
    print(f"[INFO] Original degree-0 isolates: {len(isolates_full)}")

    # Build weighted edges (only positive)
    nodes_full = list(G_full.nodes())
    node_index_full = {n: i for i, n in enumerate(nodes_full)}

    edges = []
    for u, v, data in G_full.edges(data=True):
        w = edge_weight_from_attrs(data, FEATURE_WEIGHTS)
        if w > 0:
            edges.append((u, v, float(w)))
    print(f"[INFO] Positive weighted edges: {len(edges)}")

    # Symmetric TOPK pruning on weighted edges
    if TOPK is not None:
        edges = symmetric_topk(edges, TOPK)
        print(f"[INFO] After TOPK={TOPK}: {len(edges)} weighted edges")

    # Define ACTIVE nodes = nodes that appear in the retained weighted edge list
    if len(edges) == 0:
        raise RuntimeError("No positive weighted edges after pruning. Cannot run GCLU.")

    active_nodes_set = set()
    for u, v, _ in edges:
        active_nodes_set.add(u)
        active_nodes_set.add(v)

    dropped_nodes = [n for n in nodes_full if n not in active_nodes_set]
    print(f"[INFO] Active nodes for clustering: {len(active_nodes_set)} / total {len(nodes_full)}")
    print(f"[INFO] Dropped nodes (no retained positive edges): {len(dropped_nodes)}")

    # Keep them completely out of downstream artifacts:
    # -> cluster ONLY on G_active, save ONLY G_active artifacts
    G = G_full.subgraph(active_nodes_set).copy()

    # Build index for ACTIVE node list
    nodes = list(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}

    # Convert pruned weighted edges to indices over ACTIVE nodes
    # (ignore any edge whose endpoint got dropped, though it shouldn't happen)
    edges_idx = []
    for u, v, w in edges:
        if u in node_index and v in node_index:
            edges_idx.append([node_index[u], node_index[v], float(w)])

    # Compress active nodes for GCLU input (some nodes might still be isolated within edges_idx due to pruning)
    src = np.array([e[0] for e in edges_idx], dtype=np.int64)
    dst = np.array([e[1] for e in edges_idx], dtype=np.int64)
    active = np.unique(np.concatenate([src, dst]))
    old2new = {old: i for i, old in enumerate(active)}
    src_c = np.vectorize(old2new.__getitem__)(src)
    dst_c = np.vectorize(old2new.__getitem__)(dst)
    print(f"[INFO] Active-in-edge nodes (GCLU input): {len(active)} / nodes_in_G_active {len(nodes)}")

    # IMPORTANT: nodes that are in G but not in (src,dst) after pruning are "isolated in pruned weighted graph".
    # To keep them OUT (as you requested), we drop them too from clustering artifacts:
    nodes_in_gclu = [nodes[i] for i in active.tolist()]
    G = G.subgraph(nodes_in_gclu).copy()
    nodes = list(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    print(f"[INFO] Final nodes after dropping pruned-weight isolates: {len(nodes)}")

    # Rebuild edges_idx over final nodes
    edges_idx = []
    for u, v, w in edges:
        if u in node_index and v in node_index:
            edges_idx.append([node_index[u], node_index[v], float(w)])

    if len(edges_idx) == 0:
        raise RuntimeError("No edges remain after final pruning. Cannot run GCLU.")

    src = np.array([e[0] for e in edges_idx], dtype=np.int64)
    dst = np.array([e[1] for e in edges_idx], dtype=np.int64)

    # Run GCLU
    try:
        from gclu import gclu
    except:
        raise ImportError("Install gclu: https://github.com/uef-machine-learning/gclu")

    edges_gclu = [[int(u), int(v), float(w)] for u, v, w in edges_idx]
    print(f"[INFO] Running GCLU on {len(nodes)} nodes, {len(edges_gclu)} edges...")
    t1 = time.time()
    labels = gclu(
        edges_gclu,
        graph_type="similarity",
        num_clusters=NUM_CLUSTERS,
        repeats=REPEATS,
        scale="log",
        seed=SEED,
        costf="inv",
    )
    t2 = time.time()
    print(f"[INFO] GCLU done in {t2 - t1:.2f}s")
    labels = np.array(labels, dtype=np.int32)

    # Normalize labels (over ACTIVE nodes only)
    labels_norm, _ = normalize_labels(labels)
    tag = f"K{labels_norm.max()+1}_topk{TOPK or 'all'}"

    # Save dropped nodes list ONLY as audit (not used downstream)
    dropped_fp = os.path.join(OUT_DIR, f"dropped_nodes_{tag}.json")
    with open(dropped_fp, "w", encoding="utf-8") as f:
        json.dump([str(n) for n in dropped_nodes], f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved dropped nodes audit -> {dropped_fp}")

    # Save labels & matrix (ACTIVE nodes only)
    labels_path = os.path.join(OUT_DIR, f"labels_{tag}.npy")
    np.save(labels_path, labels_norm)
    matrix_info = save_cluster_matrix(labels_norm, nodes, OUT_DIR, tag)

    # Build clusters (use node IDs, not str, to match G)
    cluster_nodes = defaultdict(list)
    for i, c in enumerate(labels_norm):
        cluster_nodes[int(c)].append(nodes[i])

    # Save manifest (ACTIVE nodes only)
    manifest = {
        "tag": tag,
        "num_clusters": int(labels_norm.max() + 1),
        "num_nodes": int(len(nodes)),
        "num_edges": int(G.number_of_edges()),
        "labels_path": os.path.abspath(labels_path),
        "matrix": matrix_info,
        "cluster_sizes": {str(k): len(v) for k, v in cluster_nodes.items()},
        "dropped_nodes_audit": os.path.abspath(dropped_fp),
        "notes": "This run excludes nodes with no retained positive weighted edges (and pruned-weight isolates) from ALL artifacts.",
    }
    manifest_path = os.path.join(OUT_DIR, f"cluster_matrix_manifest_{tag}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved manifest -> {manifest_path}")

    # Save graph with cluster labels (ACTIVE nodes only)
    for i, node in enumerate(nodes):
        G.nodes[node]["cluster"] = int(labels_norm[i])
    graph_clustered_path = os.path.join(OUT_DIR, f"graph_with_clusters_{tag}.pkl")
    with open(graph_clustered_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] Saved clustered graph -> {graph_clustered_path}")

    # --- Medoid + distance-to-center (ACTIVE nodes only) ---
    # Save node order (ACTIVE nodes only)
    nodes_order_path = os.path.join(OUT_DIR, f"nodes_order_{tag}.txt")
    with open(nodes_order_path, "w", encoding="utf-8") as f:
        for n in nodes:
            f.write(str(n) + "\n")
    print(f"[INFO] Saved nodes order -> {nodes_order_path}")

    medoids, medoid_dists = find_cluster_medoids_unweighted(G, cluster_nodes)

    # Save medoids
    centers_path = os.path.join(OUT_DIR, f"cluster_centers_{tag}.json")
    with open(centers_path, "w", encoding="utf-8") as f:
        json.dump({int(cid): str(n) for cid, n in medoids.items()}, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved centers -> {centers_path}")

    # Build & save global distance vector (ACTIVE nodes only; aligned to nodes_order)
    dist_vec = build_distance_vector(nodes, medoids, medoid_dists, unreachable=-1)
    dist_base = os.path.join(OUT_DIR, f"cluster_distances_{tag}")
    _ = try_save_tensor(dist_vec, dist_base)

    print("[DONE] Clustering complete (isolated/inactive nodes excluded from artifacts).")

if __name__ == "__main__":
    main()
