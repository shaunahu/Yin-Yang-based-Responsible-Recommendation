#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export per-node distance-to-center (unweighted shortest path) for each cluster.

Usage examples:
  # fast path: reuse saved artifacts from run_gclu_from_graph.py
  python3 graph_clustering/export_cluster_distances.py --tag K5_topk5

  # if artifacts are missing, recompute from the clustered graph .pkl
  python3 graph_clustering/export_cluster_distances.py --tag K5_topk5 --recompute

  # also save a tensor aligned to nodes_order_{tag}.txt
  python3 graph_clustering/export_cluster_distances.py --tag K5_topk5 --save-tensor
"""

import os
import json
import argparse
import pickle
from collections import defaultdict

import numpy as np
import networkx as nx
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "saved_clusters"))

def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def recompute_per_cluster_distances_csv_friendly(G, centers):
    """
    For each cluster center node, run unweighted SSSP (BFS) on G.
    For nodes in the same cluster but unreachable from the center, replace -1 with:
        penalty = (max reachable distance within that cluster) + 1

    Returns: rows = list of dicts with {cluster, node, distance_to_center}
    """
    # cluster -> node list from node attribute "cluster"
    cluster_nodes = defaultdict(list)
    for n in G.nodes():
        cid = G.nodes[n].get("cluster")
        if cid is None:
            continue
        cluster_nodes[int(cid)].append(n)

    rows = []
    for cid_raw, center in centers.items():
        cid = int(cid_raw)
        if center is None:
            continue
        if cid not in cluster_nodes:
            continue

        nodes_in_cluster = list(cluster_nodes[cid])

        # BFS from the chosen center
        lengths = nx.single_source_shortest_path_length(G, source=center)

        # collect reachable distances inside this cluster
        reachable = []
        for n in nodes_in_cluster:
            d = lengths.get(n, None)
            if d is not None:
                reachable.append(int(d))

        # define penalty distance = max+1 (if nothing reachable, fallback to 1)
        penalty = (max(reachable) + 1) if len(reachable) > 0 else 1

        # output rows: replace unreachable with penalty
        for n in nodes_in_cluster:
            d = lengths.get(n, None)
            if d is None:
                rows.append({"cluster": cid, "node": str(n), "distance_to_center": penalty})
            else:
                rows.append({"cluster": cid, "node": str(n), "distance_to_center": int(d)})

    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="e.g., K5_topk5")
    ap.add_argument("--recompute", action="store_true",
                    help="Force recompute distances from clustered graph .pkl instead of using pre-saved distances.")
    ap.add_argument("--save-tensor", action="store_true",
                    help="Also save a tensor aligned to nodes_order_{tag}.txt (unchanged behavior).")
    args = ap.parse_args()

    tag = args.tag

    nodes_order_fp   = os.path.join(OUT_DIR, f"nodes_order_{tag}.txt")
    centers_json_fp  = os.path.join(OUT_DIR, f"cluster_centers_{tag}.json")
    graph_pkl_fp     = os.path.join(OUT_DIR, f"graph_with_clusters_{tag}.pkl")
    dist_tensor_base = os.path.join(OUT_DIR, f"cluster_distances_{tag}")  # .pt or .npy
    csv_out_fp       = os.path.join(OUT_DIR, f"cluster_node_distances_{tag}.csv")

    # ---- Fast path (kept as-is): expands precomputed vector to rows.
    # NOTE: This can still contain -1 if your precomputed tensor has -1.
    # If you want CSV-friendly replacement, use --recompute so we can compute penalties per cluster.
    if (not args.recompute and
        os.path.exists(dist_tensor_base + ".pt") and
        os.path.exists(nodes_order_fp) and
        os.path.exists(centers_json_fp)):
        try:
            import torch
            dist_vec = torch.load(dist_tensor_base + ".pt", map_location="cpu").numpy()
            with open(nodes_order_fp, "r", encoding="utf-8") as f:
                nodes_order = [line.strip() for line in f]
            with open(centers_json_fp, "r", encoding="utf-8") as f:
                centers = json.load(f)  # {cluster_id: center_node}

            if not os.path.exists(graph_pkl_fp):
                raise FileNotFoundError(f"Missing graph file needed to map node->cluster: {graph_pkl_fp}")
            G = load_graph(graph_pkl_fp)

            rows = []
            for idx, n in enumerate(nodes_order):
                cid = G.nodes[n].get("cluster", None)
                if cid is None:
                    continue
                d = int(dist_vec[idx])
                rows.append({"cluster": int(cid), "node": str(n), "distance_to_center": d})

            pd.DataFrame(rows).to_csv(csv_out_fp, index=False)
            print(f"✅ Wrote: {csv_out_fp}  (from precomputed tensor)")
            print("ℹ️ If you want to replace -1 with per-cluster (max+1) for CSV, run with --recompute.")
            if args.save_tensor:
                print(f"ℹ️ Tensor already exists: {dist_tensor_base}.pt")
            return
        except Exception as e:
            print(f"[WARN] Fast path failed ({e}). Falling back to recompute.")
            args.recompute = True

    # ---- Recompute path (CSV-friendly): BFS + per-cluster penalty replacement
    if not os.path.exists(graph_pkl_fp):
        raise FileNotFoundError(f"Clustered graph not found: {graph_pkl_fp}")
    if not os.path.exists(centers_json_fp):
        raise FileNotFoundError(f"Cluster centers JSON not found: {centers_json_fp}")

    with open(centers_json_fp, "r", encoding="utf-8") as f:
        centers = json.load(f)

    G = load_graph(graph_pkl_fp)

    rows = recompute_per_cluster_distances_csv_friendly(G, centers)
    df = pd.DataFrame(rows, columns=["cluster", "node", "distance_to_center"])
    df.sort_values(["cluster", "distance_to_center", "node"], inplace=True, kind="mergesort")
    df.to_csv(csv_out_fp, index=False)
    print(f"✅ Wrote (CSV-friendly, no -1): {csv_out_fp}")

    # Optional: save tensor aligned to nodes_order (unchanged; still uses -1 if unreachable)
    if args.save_tensor:
        if not os.path.exists(nodes_order_fp):
            nodes_order = list(G.nodes())
            with open(nodes_order_fp, "w", encoding="utf-8") as f:
                for n in nodes_order:
                    f.write(str(n) + "\n")
        else:
            with open(nodes_order_fp, "r", encoding="utf-8") as f:
                nodes_order = [line.strip() for line in f]

        dist_map = dict(zip(df["node"].astype(str).tolist(),
                            df["distance_to_center"].astype(int).tolist()))
        vec = np.array([dist_map.get(str(n), -1) for n in nodes_order], dtype=np.int64)

        try:
            import torch
            torch.save(torch.from_numpy(vec), dist_tensor_base + ".pt")
            print(f"✅ Saved tensor: {dist_tensor_base}.pt  (aligned to {os.path.basename(nodes_order_fp)})")
        except Exception as e:
            np.save(dist_tensor_base + ".npy", vec)
            print(f"ℹ️ Torch unavailable; saved numpy: {dist_tensor_base}.npy  ({e})")

if __name__ == "__main__":
    main()