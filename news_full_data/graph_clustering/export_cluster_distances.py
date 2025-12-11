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

def recompute_per_cluster_distances(G, centers):
    """
    For each cluster center node, run unweighted SSSP to its cluster.
    Returns: rows = list of dicts with {cluster, node, distance_to_center}
    """
    # Build cluster -> items from node attribute "cluster"
    cluster_nodes = defaultdict(list)
    for n in G.nodes():
        cid = G.nodes[n].get("cluster")
        if cid is None:
            continue
        cluster_nodes[int(cid)].append(n)

    rows = []
    for cid, center in centers.items():
        if center is None:
            continue
        nodes_in_cluster = set(cluster_nodes[int(cid)])
        # single-source unweighted shortest path lengths
        lengths = nx.single_source_shortest_path_length(G, source=center)
        # keep only nodes of this cluster
        for n in nodes_in_cluster:
            d = lengths.get(n, None)
            if d is None:
                # unreachable from center inside the same connected component
                rows.append({"cluster": int(cid), "node": str(n), "distance_to_center": -1})
            else:
                rows.append({"cluster": int(cid), "node": str(n), "distance_to_center": int(d)})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="e.g., K5_topk5")
    ap.add_argument("--recompute", action="store_true",
                    help="Force recompute distances from clustered graph .pkl instead of using pre-saved distances.")
    ap.add_argument("--save-tensor", action="store_true",
                    help="Also save a tensor aligned to nodes_order_{tag}.txt")
    args = ap.parse_args()

    tag = args.tag
    # expected files
    nodes_order_fp   = os.path.join(OUT_DIR, f"nodes_order_{tag}.txt")
    centers_json_fp  = os.path.join(OUT_DIR, f"cluster_centers_{tag}.json")
    graph_pkl_fp     = os.path.join(OUT_DIR, f"graph_with_clusters_{tag}.pkl")
    dist_tensor_base = os.path.join(OUT_DIR, f"cluster_distances_{tag}")  # .pt or .npy
    csv_out_fp       = os.path.join(OUT_DIR, f"cluster_node_distances_{tag}.csv")

    # Prefer fast path: load precomputed distances vector and expand to rows
    if not args.recompute and os.path.exists(dist_tensor_base + ".pt") and os.path.exists(nodes_order_fp) and os.path.exists(centers_json_fp):
        try:
            import torch
            dist_vec = torch.load(dist_tensor_base + ".pt", map_location="cpu").numpy()
            with open(nodes_order_fp, "r", encoding="utf-8") as f:
                nodes_order = [line.strip() for line in f]
            with open(centers_json_fp, "r", encoding="utf-8") as f:
                centers = json.load(f)  # {cluster_id: center_node}
            # Need cluster id per node → read from graph (lightweight)
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
            if args.save_tensor:
                # Already have tensor; just confirm its path
                print(f"ℹ️ Tensor already exists: {dist_tensor_base}.pt")
            return
        except Exception as e:
            print(f"[WARN] Fast path failed ({e}). Falling back to recompute.")
            args.recompute = True

    # Recompute path: load graph & centers, then BFS per center
    if not os.path.exists(graph_pkl_fp):
        raise FileNotFoundError(f"Clustered graph not found: {graph_pkl_fp}")

    if os.path.exists(centers_json_fp):
        with open(centers_json_fp, "r", encoding="utf-8") as f:
            centers = json.load(f)
    else:
        # If centers not present, choose medoid as the highest-degree node per cluster (simple, fast fallback)
        print("[INFO] centers JSON missing; choosing degree-based centers per cluster (fallback).")
        G_tmp = load_graph(graph_pkl_fp)
        cluster_nodes = defaultdict(list)
        for n in G_tmp.nodes():
            cid = G_tmp.nodes[n].get("cluster")
            if cid is not None:
                cluster_nodes[int(cid)].append(n)
        centers = {}
        for cid, ns in cluster_nodes.items():
            centers[int(cid)] = max(ns, key=lambda x: G_tmp.degree(x))

    G = load_graph(graph_pkl_fp)
    rows = recompute_per_cluster_distances(G, centers)
    df = pd.DataFrame(rows, columns=["cluster", "node", "distance_to_center"])
    df.sort_values(["cluster", "distance_to_center", "node"], inplace=True, kind="mergesort")
    df.to_csv(csv_out_fp, index=False)
    print(f"✅ Wrote: {csv_out_fp}")

    # Optional: save a tensor aligned to nodes_order
    if args.save_tensor:
        if not os.path.exists(nodes_order_fp):
            # build nodes_order from the graph for alignment
            nodes_order = list(G.nodes())
            with open(nodes_order_fp, "w", encoding="utf-8") as f:
                for n in nodes_order:
                    f.write(str(n) + "\n")
        else:
            with open(nodes_order_fp, "r", encoding="utf-8") as f:
                nodes_order = [line.strip() for line in f]

        # map node -> distance from the CSV we just made
        dist_map = dict(zip(df["node"].astype(str).tolist(), df["distance_to_center"].astype(int).tolist()))
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
