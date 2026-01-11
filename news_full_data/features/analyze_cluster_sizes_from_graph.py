#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze how many nodes are in each cluster from:
  newsGraph/graph_with_clusters_K5_topk5_news.pkl

It tries to auto-detect the node attribute that stores cluster id, e.g.:
  cluster, cluster_id, label, community, partition, etc.

Outputs:
  - prints cluster sizes
  - saved_clusters/cluster_sizes_from_graph_{tag}.json
  - saved_clusters/cluster_sizes_from_graph_{tag}.csv

Run:
  python features/analyze_cluster_sizes_from_graph.py
  python features/analyze_cluster_sizes_from_graph.py --tag K5_topk5
  python features/analyze_cluster_sizes_from_graph.py --attr cluster
"""

import os
import json
import csv
import argparse
import pickle
from collections import Counter
from typing import Optional, Tuple, Dict, Any

import networkx as nx


# -------------------------
# Hard-coded repo layout
# -------------------------
HERE = os.path.dirname(os.path.abspath(__file__))          # .../news_full_data/features
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))  # .../news_full_data

NEWS_GRAPH_DIR = os.path.join(PROJECT_ROOT, "newsGraph")
SAVED_CLUSTERS_DIR = os.path.join(PROJECT_ROOT, "saved_clusters")


DEFAULTS = dict(
    tag="K5_topk5",
    graph_pkl=os.path.join(NEWS_GRAPH_DIR, "graph_pruned_K5_topk5_news.pkl"),
)


CANDIDATE_ATTRS = [
    "cluster", "cluster_id", "clusterid",
    "label", "labels",
    "community", "community_id",
    "partition", "part",
    "group", "group_id",
    "cid",
]


def load_graph(pkl_path: str) -> nx.Graph:
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Graph pkl not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Loaded object is not a NetworkX graph: {type(obj)}")
    return obj


def detect_cluster_attr(G: nx.Graph) -> Tuple[Optional[str], Dict[str, int]]:
    """
    Try to detect which node attribute stores cluster id.
    Returns (attr_name_or_None, coverage_dict attr->count_of_nodes_having_it)
    """
    coverage = {a: 0 for a in CANDIDATE_ATTRS}
    # quick scan up to some nodes
    for i, (n, data) in enumerate(G.nodes(data=True)):
        for a in CANDIDATE_ATTRS:
            if a in data and data[a] is not None:
                coverage[a] += 1
        if i >= 5000:  # enough for detection
            break

    # choose the attr with max coverage
    best_attr = None
    best_cov = 0
    for a, c in coverage.items():
        if c > best_cov:
            best_attr = a
            best_cov = c

    # sanity: require at least some nodes
    if best_cov == 0:
        return None, coverage
    return best_attr, coverage


def coerce_cluster_id(v: Any):
    """
    Normalize cluster id to int/string if possible for stable counting.
    """
    if v is None:
        return None
    # torch/numpy scalar -> python scalar
    try:
        if hasattr(v, "item"):
            v = v.item()
    except Exception:
        pass
    # cast numeric-like strings to int
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            try:
                return int(s)
            except Exception:
                return s
        return s
    # cast floats that are basically ints
    if isinstance(v, float):
        if abs(v - round(v)) < 1e-9:
            return int(round(v))
    return v


def count_cluster_sizes(G: nx.Graph, attr: str) -> Counter:
    cnt = Counter()
    missing = 0
    for n, data in G.nodes(data=True):
        if attr not in data or data[attr] is None:
            missing += 1
            continue
        cid = coerce_cluster_id(data[attr])
        if cid is None:
            missing += 1
            continue
        cnt[cid] += 1

    if missing > 0:
        print(f"[WARN] {missing} nodes missing '{attr}' attribute (ignored in counts).")
    return cnt


def save_outputs(tag: str, counter: Counter):
    os.makedirs(SAVED_CLUSTERS_DIR, exist_ok=True)
    # sort by cluster id (int clusters first), otherwise by string
    def sort_key(k):
        return (0, k) if isinstance(k, int) else (1, str(k))

    items = sorted(counter.items(), key=lambda x: sort_key(x[0]))

    out_json = os.path.join(SAVED_CLUSTERS_DIR, f"cluster_sizes_from_graph_{tag}.json")
    out_csv = os.path.join(SAVED_CLUSTERS_DIR, f"cluster_sizes_from_graph_{tag}.csv")

    # json (keys as str, to match your manifest style)
    data = {str(k): int(v) for k, v in items}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tag": tag,
                "cluster_sizes": data,
                "num_clusters_observed": len(data),
                "num_nodes_counted": sum(data.values()),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # csv
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "num_nodes"])
        for k, v in items:
            w.writerow([k, int(v)])

    print(f"[SAVE] {out_json}")
    print(f"[SAVE] {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=DEFAULTS["tag"])
    ap.add_argument("--graph-pkl", default=DEFAULTS["graph_pkl"])
    ap.add_argument(
        "--attr",
        default=None,
        help="Force the node attribute name for cluster id (skip auto-detect).",
    )
    args = ap.parse_args()

    print("[PATHS]")
    print("GRAPH_PKL :", os.path.abspath(args.graph_pkl))

    G = load_graph(args.graph_pkl)

    if args.attr is None:
        attr, coverage = detect_cluster_attr(G)
        print("[DETECT] candidate coverage (sampled):")
        for k, v in sorted(coverage.items(), key=lambda x: -x[1]):
            if v > 0:
                print(f"  - {k}: {v}")
        if attr is None:
            # print a hint: show some node keys
            n0, d0 = next(iter(G.nodes(data=True)))
            print("[ERROR] Cannot detect cluster attribute automatically.")
            print(f"Example node: {n0}")
            print(f"Node attribute keys: {sorted(list(d0.keys()))}")
            print("Please rerun with:  --attr <your_cluster_attr_name>")
            return
        print(f"[DETECT] using attr = '{attr}'")
    else:
        attr = args.attr
        print(f"[FORCE] using attr = '{attr}'")

    counter = count_cluster_sizes(G, attr=attr)

    # print results
    total = sum(counter.values())
    print("\n[CLUSTER SIZES]")
    for cid, c in sorted(counter.items(), key=lambda x: (0, x[0]) if isinstance(x[0], int) else (1, str(x[0]))):
        print(f"  cluster {cid}: {c}")
    print(f"\n[SUMMARY] counted nodes = {total} / total nodes in graph = {G.number_of_nodes()}")
    print(f"[SUMMARY] edges in graph = {G.number_of_edges()}")

    save_outputs(args.tag, counter)


if __name__ == "__main__":
    main()
