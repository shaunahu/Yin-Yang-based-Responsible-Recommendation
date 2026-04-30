#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
import numpy as np
import torch


# =========================================================
# PATHS
# =========================================================
HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(HERE)

DATA_DIR = os.path.join(BASE_DIR, "A_graph_data", "K5_weighted_node2vec_v4")

GRAPH_PATH = os.path.join(DATA_DIR, "graph_with_clusters.pkl")
EMB_PATH = os.path.join(DATA_DIR, "node_embeddings.pkl")
CENTROID_PATH = os.path.join(DATA_DIR, "cluster_centroids.json")

PT_OUT = os.path.join(DATA_DIR, "node_to_center_distance.pt")
JSON_OUT = os.path.join(DATA_DIR, "node_to_center_distance.json")
NODE_ORDER_OUT = os.path.join(DATA_DIR, "node_order_for_distance.json")


# =========================================================
# HELPERS
# =========================================================
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def euclidean_distance(x, y):
    return float(np.linalg.norm(x - y))


# =========================================================
# MAIN
# =========================================================
def main():
    print("[INFO] Loading data...")

    G = load_pickle(GRAPH_PATH)
    node_embeddings_raw = load_pickle(EMB_PATH)
    centroids_raw = load_json(CENTROID_PATH)

    # convert to numpy
    node_embeddings = {
        str(k): np.asarray(v, dtype=np.float32)
        for k, v in node_embeddings_raw.items()
    }

    centroids = {
        str(k): np.asarray(v, dtype=np.float32)
        for k, v in centroids_raw.items()
    }

    nodes = list(G.nodes())

    results = {}
    cluster_max = {}

    print("[INFO] Computing distances...")

    # ===== STEP 1: compute raw distances =====
    for node in nodes:
        node_str = str(node)

        if node_str not in node_embeddings:
            continue

        if "cluster" not in G.nodes[node]:
            continue

        cluster_id = str(G.nodes[node]["cluster"])

        if cluster_id not in centroids:
            continue

        emb = node_embeddings[node_str]
        centroid = centroids[cluster_id]

        d = euclidean_distance(emb, centroid)

        results[node_str] = {
            "cluster": int(cluster_id),
            "distance_to_center": d
        }

        # track max per cluster
        if cluster_id not in cluster_max:
            cluster_max[cluster_id] = d
        else:
            cluster_max[cluster_id] = max(cluster_max[cluster_id], d)

    print("[INFO] Normalizing per cluster...")

    # ===== STEP 2: normalize (0~1 per cluster) =====
    for node, info in results.items():
        c = str(info["cluster"])
        d = info["distance_to_center"]

        max_d = cluster_max[c] if cluster_max[c] > 0 else 1.0
        d_norm = d / max_d

        info["distance_norm"] = float(d_norm)

    # ===== STEP 3: build tensor =====
    node_order = list(results.keys())

    dist_norm_vector = np.array(
        [results[n]["distance_norm"] for n in node_order],
        dtype=np.float32
    )

    dist_tensor = torch.from_numpy(dist_norm_vector)

    # ===== STEP 4: save =====
    print("[INFO] Saving...")

    torch.save(dist_tensor, PT_OUT)

    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(NODE_ORDER_OUT, "w", encoding="utf-8") as f:
        json.dump(node_order, f, indent=2, ensure_ascii=False)

    # ===== STATS =====
    print("[INFO] Done.")
    print(f"[SAVE] PT   -> {PT_OUT}")
    print(f"[SAVE] JSON -> {JSON_OUT}")
    print(f"[SAVE] ORDER-> {NODE_ORDER_OUT}")

    print("[INFO] Distance norm stats:")
    print(f"  min  = {dist_norm_vector.min():.6f}")
    print(f"  max  = {dist_norm_vector.max():.6f}")
    print(f"  mean = {dist_norm_vector.mean():.6f}")
    print(f"  std  = {dist_norm_vector.std():.6f}")


if __name__ == "__main__":
    main()