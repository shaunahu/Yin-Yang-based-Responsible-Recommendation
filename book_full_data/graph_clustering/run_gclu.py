#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run_gclu_from_graph.py
#
# Cluster from a NetworkX graph_with_edges.pkl (MOVIES), save:
# - labels (.npy)
# - cluster membership matrix (.pt if torch, else .npy)
# - manifest JSON (includes cluster_to_items for downstream belief script)
# - clustered graph as .pkl (node attribute: "cluster")
# - cluster medoids (JSON) + distance-to-center vector (.pt if torch, else .npy)
#
# Outputs are saved to: ../saved_clusters
# Input graph default:    ../moviesGraph/graph_with_edges.pkl

import os
import json
import time
import heapq
import pickle
import random
import argparse
import numpy as np
from collections import defaultdict
import networkx as nx

# ---------------------------
# Defaults (MOVIES dataset)
# ---------------------------
DEFAULT_FEATURE_WEIGHTS = {
    "semantic_similarity": 0.1,
    "topic_similarity": 0.7,
    "sentiment_similarity": 0.1,
    "cooccurrence": 0.1,
}
FEATURE_ALIASES = {
    "cooccurrence": ["cooccurrence", "frequent", "impression_cooccurrence_prob"],
    "semantic_similarity": ["semantic_similarity", "semantic_similarity_ta"],
    "topic_similarity": ["topic_similarity"],
    "sentiment_similarity": ["sentiment_similarity"],
}

DEFAULT_TOPK = 5
DEFAULT_NUM_CLUSTERS = 5
DEFAULT_REPEATS = 30
DEFAULT_SEED = 123

# Medoid config
MEDOID_EXACT_THRESHOLD = 4000   # exact BFS per candidate if cluster size <= threshold
MEDOID_SAMPLES_LARGE = 256      # otherwise sample this many candidate centers


# ---------------------------
# Utility
# ---------------------------
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
            except Exception:
                pass
    return float(w)

def symmetric_topk(edge_list, k: int):
    """
    edge_list: list[(u_node, v_node, weight)]
    Keep top-k edges per node (symmetric), by weight.
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
    """
    Save cluster membership matrix KxN as bool.
    Prefer torch .pt; fallback to numpy .npy.
    """
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
        return {"format": "pt", "path": os.path.abspath(path), "shape": [K, N], "dtype": "bool"}
    except Exception:
        M = np.zeros((K, N), bool)
        for i, c in enumerate(labels):
            M[int(c), i] = True
        path = os.path.join(out_dir, f"cluster_matrix_{tag}.npy")
        np.save(path, M)
        return {"format": "npy", "path": os.path.abspath(path), "shape": [K, N], "dtype": "bool"}


# ---------------------------
# Medoid & distance helpers
# ---------------------------
def _bfs_dist_sum(G, cluster_set, source):
    """
    Unweighted BFS shortest-path lengths from source.
    Sum distances for nodes in cluster_set.
    """
    dist_map = {}
    lengths = nx.single_source_shortest_path_length(G, source)
    for n, d in lengths.items():
        if n in cluster_set:
            dist_map[n] = d
    total = sum(dist_map.values())
    return total, dist_map

def find_cluster_medoids_unweighted(G, cluster_nodes, seed=DEFAULT_SEED):
    RNG = random.Random(seed)
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
        return {"path": os.path.abspath(path), "format": "pt", "shape": list(t.shape)}
    except Exception:
        npy = f"{out_base}.npy"
        np.save(npy, dist_vec)
        print(f"[SAVE] distances -> {npy}")
        return {"path": os.path.abspath(npy), "format": "npy", "shape": list(dist_vec.shape)}


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    default_graph_pkl = os.path.abspath(os.path.join(here, "..", "booksGraph", "graph_with_edges.pkl"))
    default_out_dir = os.path.abspath(os.path.join(here, "..", "saved_clusters"))

    p = argparse.ArgumentParser(
        description="Cluster from NetworkX graph_with_edges.pkl using GCLU; save outputs to saved_clusters/"
    )
    p.add_argument("--graph_pkl", default=default_graph_pkl, help="Path to graph_with_edges.pkl")
    p.add_argument("--out_dir", default=default_out_dir, help="Output directory (default: ../saved_clusters)")
    p.add_argument("--topk", type=int, default=DEFAULT_TOPK, help="Symmetric top-k pruning (<=0 disables)")
    p.add_argument("--num_clusters", type=int, default=DEFAULT_NUM_CLUSTERS, help="Number of clusters (K)")
    p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="GCLU repeats")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    p.add_argument(
        "--feature_weights_json",
        default="",
        help=("Optional: JSON string or JSON file path overriding feature weights. "
              "Example: '{\"semantic_similarity\":0.7,\"cooccurrence\":0.3}'")
    )
    return p.parse_args()

def load_feature_weights(arg: str):
    if not arg:
        return dict(DEFAULT_FEATURE_WEIGHTS)

    # If arg is a file path
    if os.path.exists(arg) and os.path.isfile(arg):
        with open(arg, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("feature_weights_json file must contain a JSON object")
        return obj

    # Otherwise parse as JSON string
    obj = json.loads(arg)
    if not isinstance(obj, dict):
        raise ValueError("feature_weights_json must be a JSON object")
    return obj


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    feature_weights = load_feature_weights(args.feature_weights_json)
    topk = args.topk if args.topk and args.topk > 0 else None

    # Load graph
    t0 = time.time()
    G = load_graph(args.graph_pkl)
    nodes = list(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    print(f"[INFO] Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, time={time.time()-t0:.2f}s")
    print(f"[INFO] Graph path: {args.graph_pkl}")
    print(f"[INFO] Out dir: {args.out_dir}")

    # Build weighted edges (keep w>0)
    edges = []
    for u, v, data in G.edges(data=True):
        w = edge_weight_from_attrs(data, feature_weights)
        if w > 0:
            edges.append([node_index[u], node_index[v], w])

    print(f"[INFO] Positive edges: {len(edges)}")
    if len(edges) == 0:
        print("[WARN] No positive edges after weighting. Check feature names/aliases vs edge attributes.")
        print(f"[WARN] feature_weights={feature_weights}")

    # Symmetric top-k pruning
    if topk is not None:
        edges_l = [(nodes[u], nodes[v], w) for u, v, w in edges]
        edges_l = symmetric_topk(edges_l, topk)
        edges = [[node_index[u], node_index[v], w] for u, v, w in edges_l]
        print(f"[INFO] After TOPK={topk}: {len(edges)} edges")

    # Compress active nodes (nodes that appear in edges)
    src = np.array([e[0] for e in edges], dtype=np.int64)
    dst = np.array([e[1] for e in edges], dtype=np.int64)
    active = np.unique(np.concatenate([src, dst]))
    old2new = {old: i for i, old in enumerate(active)}
    src_c = np.vectorize(old2new.__getitem__)(src)
    dst_c = np.vectorize(old2new.__getitem__)(dst)
    print(f"[INFO] Active nodes: {len(active)} / total {len(nodes)}")

    # Run GCLU
    try:
        from gclu import gclu
    except Exception as e:
        raise ImportError(
            "Missing dependency 'gclu'. Install it (e.g., pip install gclu or from source): "
            "https://github.com/uef-machine-learning/gclu"
        ) from e

    weights_arr = np.array([e[2] for e in edges], dtype=np.float64)
    edges_gclu = [[int(u), int(v), float(w)] for u, v, w in zip(src_c, dst_c, weights_arr)]

    print(f"[INFO] Running GCLU... (K={args.num_clusters}, repeats={args.repeats}, seed={args.seed})")
    t1 = time.time()
    labels_active = gclu(
        edges_gclu,
        graph_type="similarity",
        num_clusters=args.num_clusters,
        repeats=args.repeats,
        scale="no",
        seed=args.seed,
        costf="inv",
    )
    t2 = time.time()
    print(f"[INFO] GCLU done in {t2 - t1:.2f}s")

    labels_active = np.array(labels_active, dtype=np.int32)

    # Expand labels back to all nodes
    labels_full = np.full(len(nodes), -1, np.int32)
    for i, a in enumerate(active):
        labels_full[a] = labels_active[i]

    # Any nodes not in active (isolated after pruning) -> assign to largest cluster
    if (labels_full == -1).any():
        unique, counts = np.unique(labels_active, return_counts=True)
        largest = int(unique[np.argmax(counts)])
        labels_full[labels_full == -1] = largest
        print(f"[INFO] Filled inactive nodes with largest cluster id={largest}")

    # Normalize labels to 0..K-1
    labels_norm, _ = normalize_labels(labels_full)
    K_final = int(labels_norm.max() + 1)
    tag = f"K{K_final}_topk{topk or 'all'}"

    # Save labels
    labels_path = os.path.join(args.out_dir, f"labels_{tag}.npy")
    np.save(labels_path, labels_norm)
    print(f"[INFO] Saved labels -> {labels_path}")

    # Save cluster matrix
    matrix_info = save_cluster_matrix(labels_norm, nodes, args.out_dir, tag)

    # Build cluster_to_items
    cluster_nodes = defaultdict(list)
    for i, c in enumerate(labels_norm):
        cluster_nodes[int(c)].append(str(nodes[i]))

    # Save clustered graph as PKL (requested)
    for i, node in enumerate(nodes):
        G.nodes[node]["cluster"] = int(labels_norm[i])

    graph_clustered_path = os.path.join(args.out_dir, f"graph_with_clusters_{tag}.pkl")
    with open(graph_clustered_path, "wb") as f:
        pickle.dump(G, f)
    print(f"[INFO] Saved clustered graph -> {graph_clustered_path}")

    # Save node order (useful for distance vector alignment)
    nodes_order_path = os.path.join(args.out_dir, f"nodes_order_{tag}.txt")
    with open(nodes_order_path, "w", encoding="utf-8") as f:
        for n in nodes:
            f.write(str(n) + "\n")

    # Save manifest (includes cluster_to_items for your belief script)
    manifest = {
        "tag": tag,
        "num_clusters": K_final,
        "num_nodes": len(nodes),
        "graph_pkl": os.path.abspath(args.graph_pkl),
        "graph_with_clusters_pkl": os.path.abspath(graph_clustered_path),
        "feature_weights": feature_weights,
        "topk": topk,
        "labels_path": os.path.abspath(labels_path),
        "matrix": matrix_info,
        "cluster_sizes": {str(k): len(v) for k, v in cluster_nodes.items()},
        "cluster_to_items": {str(k): v for k, v in cluster_nodes.items()},
        "nodes_order_path": os.path.abspath(nodes_order_path),
    }

    manifest_path = os.path.join(args.out_dir, f"cluster_matrix_manifest_{tag}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved manifest -> {manifest_path}")

    # --- Medoid + distance-to-center ---
    medoids, medoid_dists = find_cluster_medoids_unweighted(G, cluster_nodes, seed=args.seed)

    centers_path = os.path.join(args.out_dir, f"cluster_centers_{tag}.json")
    with open(centers_path, "w", encoding="utf-8") as f:
        json.dump({str(cid): str(n) for cid, n in medoids.items()}, f, indent=2, ensure_ascii=False)

    dist_vec = build_distance_vector(nodes, medoids, medoid_dists)
    dist_base = os.path.join(args.out_dir, f"cluster_distances_{tag}")
    dist_info = try_save_tensor(dist_vec, dist_base)

    # Update manifest with medoid/dist paths (overwrite manifest once)
    manifest["cluster_centers_path"] = os.path.abspath(centers_path)
    manifest["cluster_distances"] = dist_info
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Updated manifest with medoid/distance info -> {manifest_path}")

    print("[DONE] Clustering + graph PKL + medoid calculation complete.")


if __name__ == "__main__":
    main()
