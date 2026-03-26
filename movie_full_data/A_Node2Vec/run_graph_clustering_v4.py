#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import pickle
from collections import defaultdict

import numpy as np
import networkx as nx
import torch
from node2vec import Node2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PATH CONFIG
# =========================================================
HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(HERE)

GRAPH_PATH = os.path.join(BASE_DIR, "moviesGraph", "graph_with_edges.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "A_graph_data", "K5_weighted_node2vec_v4")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# CORE CONFIG
# =========================================================

EDGE_FEATURES = [
    "cooccurrence",
    "sentiment_similarity",
    "topic_similarity",
    "semantic_similarity",
]

LOWER_PERCENTILE = 1
UPPER_PERCENTILE = 99

# v4 新增参数
EDGE_WEIGHT_THRESHOLD = 0.05
TOPK_PRUNING = 5

EMBED_DIM = 128
NUM_CLUSTERS = 5
TOP_REPRESENTATIVES = 10

NODE2VEC_PARAMS = dict(
    dimensions=EMBED_DIM,
    walk_length=80,
    num_walks=10,
    p=1.0,
    q=1.0,
    workers=8,
)
NODE2VEC_WINDOW = 10

KMEANS_PARAMS = dict(
    n_clusters=NUM_CLUSTERS,
    random_state=42,
    n_init=10,
    max_iter=300,
)


# =========================================================
# HELPERS
# =========================================================
def safe_json_dump(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def robust_minmax(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    lo = np.percentile(values, lower)
    hi = np.percentile(values, upper)
    clipped = np.clip(values, lo, hi)

    if hi - lo == 0:
        return np.zeros_like(clipped, dtype=np.float32)

    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def build_cluster_matrix(labels: np.ndarray, num_clusters: int) -> torch.Tensor:
    n = len(labels)
    mat = torch.zeros((num_clusters, n), dtype=torch.bool)
    for i, c in enumerate(labels):
        mat[int(c), i] = True
    return mat


# =========================================================
# LOAD GRAPH
# =========================================================
def load_graph(path: str) -> nx.Graph:
    with open(path, "rb") as f:
        G = pickle.load(f)

    if not isinstance(G, (nx.Graph, nx.DiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")

    return G


# =========================================================
# STEP 1: KEEP LARGEST CONNECTED COMPONENT
# =========================================================
def keep_largest_connected_component(G: nx.Graph):
    if G.is_directed():
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))

    if not components:
        raise RuntimeError("Graph has no connected components.")

    sizes = sorted((len(c) for c in components), reverse=True)
    largest_cc = max(components, key=len)

    G_lcc = G.subgraph(largest_cc).copy()

    stats = {
        "num_components_before_lcc": int(len(components)),
        "top_10_component_sizes": [int(x) for x in sizes[:10]],
        "largest_component_size": int(len(largest_cc)),
        "nodes_before_lcc": int(G.number_of_nodes()),
        "edges_before_lcc": int(G.number_of_edges()),
        "nodes_after_lcc": int(G_lcc.number_of_nodes()),
        "edges_after_lcc": int(G_lcc.number_of_edges()),
    }
    return G_lcc, stats


# =========================================================
# STEP 2: NORMALIZE EDGE FEATURES + BUILD EDGE WEIGHT
# =========================================================
def compute_edge_weights(G: nx.Graph):
    edge_list = list(G.edges())
    if not edge_list:
        raise RuntimeError("Graph has no edges.")

    raw_feature_values = {feat: [] for feat in EDGE_FEATURES}

    for u, v in edge_list:
        data = G[u][v]
        for feat in EDGE_FEATURES:
            if feat not in data:
                raise KeyError(f"Edge ({u}, {v}) missing feature '{feat}'")
            raw_feature_values[feat].append(float(data[feat]))

    normalized_feature_values = {}
    audit = {
        "raw_feature_stats": {},
        "normalized_feature_stats": {},
    }

    for feat in EDGE_FEATURES:
        arr = np.array(raw_feature_values[feat], dtype=np.float32)
        norm = robust_minmax(arr, LOWER_PERCENTILE, UPPER_PERCENTILE)
        normalized_feature_values[feat] = norm

        audit["raw_feature_stats"][feat] = {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p_lower": float(np.percentile(arr, LOWER_PERCENTILE)),
            "p_upper": float(np.percentile(arr, UPPER_PERCENTILE)),
        }
        audit["normalized_feature_stats"][feat] = {
            "min": float(norm.min()),
            "max": float(norm.max()),
            "mean": float(norm.mean()),
            "std": float(norm.std()),
        }

    combined_weights = []
    for i, (u, v) in enumerate(edge_list):
        vals = [normalized_feature_values[feat][i] for feat in EDGE_FEATURES]
        weight = float(np.mean(vals))
        G[u][v]["weight"] = weight
        combined_weights.append(weight)

    combined_arr = np.array(combined_weights, dtype=np.float32)
    audit["combined_weight_stats"] = {
        "min": float(combined_arr.min()),
        "max": float(combined_arr.max()),
        "mean": float(combined_arr.mean()),
        "std": float(combined_arr.std()),
    }

    return G, audit


# =========================================================
# STEP 3: DROP WEAK EDGES
# =========================================================
def apply_edge_weight_threshold(G: nx.Graph, threshold: float):
    G2 = G.copy()
    to_remove = [(u, v) for u, v, d in G2.edges(data=True) if float(d.get("weight", 0.0)) < threshold]
    G2.remove_edges_from(to_remove)

    stats = {
        "edge_weight_threshold": float(threshold),
        "edges_removed_by_threshold": int(len(to_remove)),
        "nodes_after_threshold": int(G2.number_of_nodes()),
        "edges_after_threshold": int(G2.number_of_edges()),
    }
    return G2, stats


# =========================================================
# STEP 4: SYMMETRIC TOP-K PRUNING
# =========================================================
def symmetric_topk_prune(G: nx.Graph, k: int):
    if k is None or k <= 0:
        stats = {
            "topk_pruning": None,
            "edges_removed_by_topk": 0,
            "nodes_after_topk": int(G.number_of_nodes()),
            "edges_after_topk": int(G.number_of_edges()),
        }
        return G.copy(), stats

    keep_edges = set()

    for u in G.nodes():
        weighted_neighbors = []
        for v in G.neighbors(u):
            w = float(G[u][v].get("weight", 0.0))
            edge_key = tuple(sorted((str(u), str(v))))
            weighted_neighbors.append((w, edge_key))

        weighted_neighbors.sort(key=lambda x: x[0], reverse=True)
        for _, edge_key in weighted_neighbors[:k]:
            keep_edges.add(edge_key)

    G2 = nx.Graph()
    G2.add_nodes_from(G.nodes(data=True))

    removed_count = 0
    for u, v, d in G.edges(data=True):
        edge_key = tuple(sorted((str(u), str(v))))
        if edge_key in keep_edges:
            G2.add_edge(u, v, **d)
        else:
            removed_count += 1

    stats = {
        "topk_pruning": int(k),
        "edges_removed_by_topk": int(removed_count),
        "nodes_after_topk": int(G2.number_of_nodes()),
        "edges_after_topk": int(G2.number_of_edges()),
    }
    return G2, stats


# =========================================================
# STEP 5: REMOVE ISOLATES
# =========================================================
def remove_isolates(G: nx.Graph):
    G2 = G.copy()
    isolates = list(nx.isolates(G2))
    G2.remove_nodes_from(isolates)

    stats = {
        "isolates_removed": int(len(isolates)),
        "nodes_after_isolate_removal": int(G2.number_of_nodes()),
        "edges_after_isolate_removal": int(G2.number_of_edges()),
    }
    return G2, stats


# =========================================================
# STEP 6: WEIGHTED NODE2VEC
# =========================================================
def run_weighted_node2vec(G: nx.Graph):
    node2vec = Node2Vec(G, weight_key="weight", **NODE2VEC_PARAMS)
    model = node2vec.fit(window=NODE2VEC_WINDOW, min_count=1)

    nodes = list(G.nodes())
    node_to_row = {str(node): i for i, node in enumerate(nodes)}

    embeddings = []
    for node in nodes:
        embeddings.append(model.wv[str(node)])

    embeddings = np.array(embeddings, dtype=np.float32)
    return nodes, node_to_row, embeddings


# =========================================================
# STEP 7: KMEANS
# =========================================================
def run_kmeans(embeddings: np.ndarray):
    km = KMeans(**KMEANS_PARAMS)
    labels = km.fit_predict(embeddings).astype(np.int64)
    centroids = km.cluster_centers_.astype(np.float32)
    inertia = float(km.inertia_)

    if len(np.unique(labels)) > 1 and len(embeddings) > len(np.unique(labels)):
        sil = float(silhouette_score(embeddings, labels))
    else:
        sil = float("nan")

    return labels, centroids, inertia, sil


# =========================================================
# ARTIFACTS
# =========================================================
def build_cluster_summary(G: nx.Graph, nodes, embeddings, labels, centroids):
    node_to_cluster = {nodes[i]: int(labels[i]) for i in range(len(nodes))}
    summary = {}

    for c in range(NUM_CLUSTERS):
        idx = np.where(labels == c)[0]
        cluster_nodes = [nodes[i] for i in idx]
        vecs = embeddings[idx]
        centroid = centroids[c]

        if len(idx) == 0:
            summary[str(c)] = {
                "size": 0,
                "centroid_norm": 0.0,
                "avg_distance_to_centroid": 0.0,
                "max_distance_to_centroid": 0.0,
                "min_distance_to_centroid": 0.0,
                "intra_cluster_edge_count": 0,
                "intra_cluster_edge_weight_sum": 0.0,
                "intra_cluster_density": 0.0,
            }
            continue

        dists = np.linalg.norm(vecs - centroid, axis=1)

        intra_edge_count = 0
        intra_edge_weight_sum = 0.0
        for u in cluster_nodes:
            for v in G.neighbors(u):
                if node_to_cluster.get(v) == c and str(u) < str(v):
                    intra_edge_count += 1
                    intra_edge_weight_sum += float(G[u][v].get("weight", 0.0))

        n = len(cluster_nodes)
        possible_edges = n * (n - 1) / 2
        density = float(intra_edge_count / possible_edges) if possible_edges > 0 else 0.0

        summary[str(c)] = {
            "size": int(len(idx)),
            "centroid_norm": float(np.linalg.norm(centroid)),
            "avg_distance_to_centroid": float(np.mean(dists)),
            "max_distance_to_centroid": float(np.max(dists)),
            "min_distance_to_centroid": float(np.min(dists)),
            "intra_cluster_edge_count": int(intra_edge_count),
            "intra_cluster_edge_weight_sum": float(intra_edge_weight_sum),
            "intra_cluster_density": density,
        }

    return summary


def build_cluster_representatives(nodes, embeddings, labels, centroids, top_k):
    reps = {}

    for c in range(NUM_CLUSTERS):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            reps[str(c)] = []
            continue

        vecs = embeddings[idx]
        centroid = centroids[c]
        dists = np.linalg.norm(vecs - centroid, axis=1)
        order = np.argsort(dists)[:top_k]

        reps[str(c)] = []
        for j in order:
            node = nodes[idx[j]]
            reps[str(c)].append({
                "node_id": str(node),
                "distance_to_centroid": float(dists[j]),
            })

    return reps


def build_cluster_graph_and_transition(G: nx.Graph, nodes, labels, centroids):
    node_to_cluster = {nodes[i]: int(labels[i]) for i in range(len(nodes))}
    cross_counts = defaultdict(int)
    cross_weight_sums = defaultdict(float)

    for u, v, d in G.edges(data=True):
        cu = node_to_cluster[u]
        cv = node_to_cluster[v]
        key = (cu, cv) if cu <= cv else (cv, cu)
        cross_counts[key] += 1
        cross_weight_sums[key] += float(d.get("weight", 0.0))

    centroid_sim = cosine_similarity(centroids)
    graph_json = {"nodes": [], "edges": []}

    cluster_sizes = {c: int(np.sum(labels == c)) for c in range(NUM_CLUSTERS)}
    for c in range(NUM_CLUSTERS):
        graph_json["nodes"].append({
            "cluster_id": int(c),
            "size": cluster_sizes[c],
        })

    transition = np.zeros((NUM_CLUSTERS, NUM_CLUSTERS), dtype=np.float32)

    for (a, b), count in cross_counts.items():
        wsum = cross_weight_sums[(a, b)]

        graph_json["edges"].append({
            "source": int(a),
            "target": int(b),
            "cross_cluster_edge_count": int(count),
            "cross_cluster_edge_weight_sum": float(wsum),
            "centroid_cosine_similarity": float(centroid_sim[a, b]),
        })

        if a == b:
            transition[a, a] += wsum
        else:
            transition[a, b] += wsum
            transition[b, a] += wsum

    row_sums = transition.sum(axis=1, keepdims=True)
    nonzero = row_sums.squeeze() > 0
    transition[nonzero] = transition[nonzero] / row_sums[nonzero]

    return graph_json, transition


def build_cluster_entropy(G: nx.Graph, nodes, labels):
    node_to_cluster = {nodes[i]: int(labels[i]) for i in range(len(nodes))}
    result = {}

    for c in range(NUM_CLUSTERS):
        cluster_nodes = [n for n in nodes if node_to_cluster[n] == c]
        if not cluster_nodes:
            result[str(c)] = {
                "neighbor_cluster_entropy": 0.0,
                "neighbor_cluster_counts": {},
            }
            continue

        counts = np.zeros(NUM_CLUSTERS, dtype=np.int64)
        for u in cluster_nodes:
            for v in G.neighbors(u):
                cv = node_to_cluster[v]
                counts[cv] += 1

        result[str(c)] = {
            "neighbor_cluster_entropy": entropy_from_counts(counts),
            "neighbor_cluster_counts": {str(i): int(counts[i]) for i in range(NUM_CLUSTERS)},
        }

    return result


def build_cluster_belief_vectors(centroids):
    return centroids.astype(np.float32)


# =========================================================
# SAVE CLUSTERED GRAPH
# =========================================================
def save_graph_with_clusters(G: nx.Graph, nodes, labels, output_path: str):
    G_clustered = G.copy()
    for i, node in enumerate(nodes):
        G_clustered.nodes[node]["cluster"] = int(labels[i])

    with open(output_path, "wb") as f:
        pickle.dump(G_clustered, f, protocol=pickle.HIGHEST_PROTOCOL)


# =========================================================
# MAIN
# =========================================================
def main():
    print("Loading graph...")
    G = load_graph(GRAPH_PATH)
    print(f"Original graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    # convert node ids to strings for consistency
    if not all(isinstance(n, str) for n in G.nodes()):
        mapping = {n: str(n) for n in G.nodes()}
        G = nx.relabel_nodes(G, mapping, copy=True)

    # Step 1: Largest Connected Component
    print("Keeping largest connected component...")
    G, lcc_stats = keep_largest_connected_component(G)
    print(f"After LCC: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    # Step 2: edge weight construction
    print("Computing edge weights...")
    G, edge_weight_audit = compute_edge_weights(G)

    # Step 3: weak-edge filter
    print(f"Applying edge weight threshold: {EDGE_WEIGHT_THRESHOLD}")
    G, threshold_stats = apply_edge_weight_threshold(G, EDGE_WEIGHT_THRESHOLD)
    print(f"After threshold: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    # Step 4: top-k pruning
    print(f"Applying symmetric top-k pruning: {TOPK_PRUNING}")
    G, topk_stats = symmetric_topk_prune(G, TOPK_PRUNING)
    print(f"After top-k: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    # Step 5: remove isolates
    print("Removing isolates after pruning...")
    G, isolate_stats = remove_isolates(G)
    print(f"After isolate removal: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        raise RuntimeError("Graph became empty after filtering.")

    # Step 6: weighted node2vec
    print("Running weighted Node2Vec...")
    nodes, node_to_row, embeddings = run_weighted_node2vec(G)
    print(f"Embeddings shape: {embeddings.shape}")

    # Step 7: kmeans
    print("Running KMeans...")
    labels, centroids, inertia, sil = run_kmeans(embeddings)
    print(f"KMeans done. inertia={inertia:.4f}, silhouette={sil}")

    # Step 8: build artifacts
    print("Building artifacts...")
    cluster_matrix = build_cluster_matrix(labels, NUM_CLUSTERS)
    cluster_summary = build_cluster_summary(G, nodes, embeddings, labels, centroids)
    # cluster_reps = build_cluster_representatives(nodes, embeddings, labels, centroids, TOP_REPRESENTATIVES)
    cluster_sim = cosine_similarity(centroids).astype(np.float32)
    cluster_graph, cluster_transition = build_cluster_graph_and_transition(G, nodes, labels, centroids)
    cluster_entropy = build_cluster_entropy(G, nodes, labels)
    cluster_belief_vectors = build_cluster_belief_vectors(centroids)

    node_labels_json = {str(nodes[i]): int(labels[i]) for i in range(len(nodes))}
    cluster_to_nodes = defaultdict(list)
    for i, c in enumerate(labels):
        cluster_to_nodes[str(int(c))].append(str(nodes[i]))

    # Step 9: save files
    print("Saving outputs...")

    np.save(os.path.join(OUTPUT_DIR, "node_embeddings.npy"), embeddings)

    with open(os.path.join(OUTPUT_DIR, "node_embeddings.pkl"), "wb") as f:
        pickle.dump(
            {str(nodes[i]): embeddings[i].tolist() for i in range(len(nodes))},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    safe_json_dump(node_to_row, os.path.join(OUTPUT_DIR, "node_to_row.json"))

    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)
    safe_json_dump(node_labels_json, os.path.join(OUTPUT_DIR, "node_labels.json"))

    np.save(os.path.join(OUTPUT_DIR, "cluster_centroids.npy"), centroids)
    safe_json_dump(
        {str(i): centroids[i].tolist() for i in range(NUM_CLUSTERS)},
        os.path.join(OUTPUT_DIR, "cluster_centroids.json"),
    )

    torch.save(cluster_matrix, os.path.join(OUTPUT_DIR, "cluster_matrix.pt"))

    safe_json_dump(dict(cluster_to_nodes), os.path.join(OUTPUT_DIR, "cluster_to_nodes.json"))
    safe_json_dump(cluster_summary, os.path.join(OUTPUT_DIR, "cluster_summary.json"))
    # safe_json_dump(cluster_reps, os.path.join(OUTPUT_DIR, "cluster_representatives.json"))
    # safe_json_dump(cluster_sim.tolist(), os.path.join(OUTPUT_DIR, "cluster_similarity.json"))
    safe_json_dump(cluster_graph, os.path.join(OUTPUT_DIR, "cluster_graph.json"))

    np.save(os.path.join(OUTPUT_DIR, "cluster_transition_matrix.npy"), cluster_transition)
    safe_json_dump(cluster_entropy, os.path.join(OUTPUT_DIR, "cluster_entropy.json"))
    np.save(os.path.join(OUTPUT_DIR, "cluster_belief_vectors.npy"), cluster_belief_vectors)

    edge_weight_config = {
        "edge_features": EDGE_FEATURES,
        "normalization": {
            "method": "robust_minmax",
            "lower_percentile": LOWER_PERCENTILE,
            "upper_percentile": UPPER_PERCENTILE,
        },
        "combination": {
            "method": "equal_mean",
            "weights": [0.25, 0.25, 0.25, 0.25],
        },
        "edge_weight_threshold": EDGE_WEIGHT_THRESHOLD,
        "topk_pruning": TOPK_PRUNING,
    }
    safe_json_dump(edge_weight_config, os.path.join(OUTPUT_DIR, "edge_weight_config.json"))
    safe_json_dump(edge_weight_audit, os.path.join(OUTPUT_DIR, "edge_weight_audit.json"))

    # NEW: save clustered graph
    graph_with_clusters_path = os.path.join(OUTPUT_DIR, "graph_with_clusters.pkl")
    save_graph_with_clusters(G, nodes, labels, graph_with_clusters_path)

    manifest = {
        "version": "v4",
        "method": "largest_connected_component + robust_minmax_edge_fusion + threshold + symmetric_topk + weighted_node2vec + kmeans",
        "graph_path": os.path.abspath(GRAPH_PATH),
        "output_dir": os.path.abspath(OUTPUT_DIR),
        "input_graph_stats": {
            "nodes_original": int(lcc_stats["nodes_before_lcc"]),
            "edges_original": int(lcc_stats["edges_before_lcc"]),
        },
        "lcc_stats": lcc_stats,
        "threshold_stats": threshold_stats,
        "topk_stats": topk_stats,
        "isolate_stats": isolate_stats,
        "final_graph_stats": {
            "nodes_final": int(G.number_of_nodes()),
            "edges_final": int(G.number_of_edges()),
        },
        "embedding": {
            "method": "weighted_node2vec",
            "dimensions": EMBED_DIM,
            "window": NODE2VEC_WINDOW,
            **NODE2VEC_PARAMS,
        },
        "clustering": {
            "method": "kmeans",
            **KMEANS_PARAMS,
        },
        "quality": {
            "kmeans_inertia": float(inertia),
            "silhouette_score": float(sil) if not math.isnan(sil) else None,
        },
        "artifacts": {
            "node_embeddings_npy": "node_embeddings.npy",
            "node_embeddings_pkl": "node_embeddings.pkl",
            "node_to_row_json": "node_to_row.json",
            "labels_npy": "labels.npy",
            "node_labels_json": "node_labels.json",
            "cluster_centroids_npy": "cluster_centroids.npy",
            "cluster_centroids_json": "cluster_centroids.json",
            "cluster_matrix_pt": "cluster_matrix.pt",
            "cluster_to_nodes_json": "cluster_to_nodes.json",
            "cluster_summary_json": "cluster_summary.json",
            # "cluster_similarity_json": "cluster_similarity.json",
            # "cluster_representatives_json": "cluster_representatives.json",
            "cluster_graph_json": "cluster_graph.json",
            "cluster_transition_matrix_npy": "cluster_transition_matrix.npy",
            "cluster_entropy_json": "cluster_entropy.json",
            "cluster_belief_vectors_npy": "cluster_belief_vectors.npy",
            "edge_weight_config_json": "edge_weight_config.json",
            "edge_weight_audit_json": "edge_weight_audit.json",
            "graph_with_clusters_pkl": "graph_with_clusters.pkl",
        },
    }
    safe_json_dump(manifest, os.path.join(OUTPUT_DIR, "manifest.json"))

    print("Done.")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()