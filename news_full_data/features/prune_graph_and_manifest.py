#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLEAN corrected script: prune ONLY nodes from cluster 0 (graph attribute),
and guarantee clusters 1–4 keep EXACTLY the same node counts as the original graph.

Source of truth for cluster membership:
  - Graph node attribute (default attr name: "cluster")
  - NOT cluster_matrix, NOT labels.npy

What it does:
1) Load graph .pkl (NetworkX)
2) Compute BEFORE cluster sizes from graph node attribute
3) Decide nodes to drop: ONLY nodes with cluster==TARGET_CLUSTER (default 0),
   and meeting prune condition:
     - mode="isolates": degree == 0
     - mode="no_positive": no incident edge with weight_attr > 0
4) Remove those nodes from graph
5) Compute AFTER cluster sizes from graph node attribute
6) ASSERT clusters other than TARGET_CLUSTER keep the same counts
7) (Optional) Prune nodes_order_{tag}.txt and cluster_matrix_{tag}.pt columns
   by removing dropped nodes (alignment uses nodes_order). This DOES NOT change
   cluster membership; it just removes dropped nodes from artifacts.

Outputs (in saved_clusters/ and newsGraph/):
- newsGraph/graph_pruned_{tag}_news.pkl
- saved_clusters/dropped_nodes_{tag}_pruned.json
- saved_clusters/nodes_order_{tag}_pruned.txt            (if nodes_order exists)
- saved_clusters/cluster_matrix_{tag}_pruned.pt          (if matrix exists)
- saved_clusters/manifest_{tag}_pruned.json              (cluster_sizes from graph truth)

Run:
  python features/prune_graph_cluster0_only.py

You can override:
  --tag K5_topk5
  --mode isolates|no_positive
  --cluster-attr cluster
  --target-cluster 0
  --weight-attr weight
"""

import os
import json
import argparse
import pickle
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
except Exception:
    torch = None

try:
    import networkx as nx
except Exception as e:
    raise RuntimeError("networkx is required. pip install networkx") from e


# =========================
# HARD-CODE PROJECT PATHS
# =========================
HERE = os.path.dirname(os.path.abspath(__file__))          # .../news_full_data/features
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))  # .../news_full_data

NEWS_GRAPH_DIR = os.path.join(PROJECT_ROOT, "newsGraph")
SAVED_CLUSTERS_DIR = os.path.join(PROJECT_ROOT, "saved_clusters")

DEFAULTS = dict(
    tag="K5_topk5",
    graph_pkl=os.path.join(NEWS_GRAPH_DIR, "graph_with_clusters_K5_topk5_news.pkl"),
    out_graph=os.path.join(NEWS_GRAPH_DIR, "graph_pruned_K5_topk5_news.pkl"),
    saved_dir=SAVED_CLUSTERS_DIR,
    out_suffix="_pruned",
    mode="no_positive",           # isolates | no_positive
    weight_attr="weight",
    cluster_attr="cluster",    # <-- IMPORTANT: this must match your graph node attribute
    target_cluster=0,          # <-- ONLY prune nodes from this cluster id
)


# -------------------------
# IO helpers
# -------------------------
def load_graph(pkl_path: str):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Graph pkl not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def save_graph(G, pkl_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(pkl_path)), exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_nodes_order(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_nodes_order(path: str, nodes: List[str]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for n in nodes:
            f.write(f"{n}\n")


def load_cluster_matrix_pt(path: str):
    if torch is None:
        raise RuntimeError("torch is required to load/prune cluster_matrix .pt. pip install torch")
    mat = torch.load(path, map_location="cpu")
    if not hasattr(mat, "shape") or len(mat.shape) != 2:
        raise ValueError(f"cluster_matrix is not 2D: {path}")
    return mat


def save_cluster_matrix_pt(mat, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(mat, path)


def normalize_node_id(x: Any) -> str:
    return str(x)


def get_node_attr(G, node, attr: str) -> Any:
    # node can be str or int; try direct, then int-cast if possible
    if node in G:
        return G.nodes[node].get(attr, None)
    try:
        node_i = int(node)
        if node_i in G:
            return G.nodes[node_i].get(attr, None)
    except Exception:
        pass
    return None


# -------------------------
# Cluster size (graph truth)
# -------------------------
def graph_cluster_sizes(G, cluster_attr: str) -> Counter:
    cnt = Counter()
    missing = 0
    for n, data in G.nodes(data=True):
        if cluster_attr not in data or data[cluster_attr] is None:
            missing += 1
            continue
        try:
            cid = int(data[cluster_attr])
        except Exception:
            cid = data[cluster_attr]
        cnt[cid] += 1
    if missing > 0:
        print(f"[WARN] {missing} nodes missing '{cluster_attr}' attribute (ignored in cluster_sizes).")
    return cnt


# -------------------------
# Prune conditions
# -------------------------
def is_isolate(G, n) -> bool:
    return G.degree(n) == 0


def has_positive_incident_edge(G, n, weight_attr: str) -> bool:
    for _, _, data in G.edges(n, data=True):
        w = data.get(weight_attr, 0.0)
        try:
            if float(w) > 0:
                return True
        except Exception:
            pass
    return False


def nodes_to_drop_only_target_cluster(
    G,
    cluster_attr: str,
    target_cluster: int,
    mode: str,
    weight_attr: str,
) -> List[Any]:
    dropped = []
    for n, data in G.nodes(data=True):
        if cluster_attr not in data or data[cluster_attr] is None:
            continue
        try:
            cid = int(data[cluster_attr])
        except Exception:
            cid = data[cluster_attr]

        if cid != target_cluster:
            continue

        if mode == "isolates":
            if is_isolate(G, n):
                dropped.append(n)
        else:
            if not has_positive_incident_edge(G, n, weight_attr=weight_attr):
                dropped.append(n)
    return dropped


# -------------------------
# Optional: prune artifacts by dropping columns corresponding to dropped nodes
# -------------------------
def prune_nodes_order_and_matrix_optional(
    saved_dir: str,
    tag: str,
    out_suffix: str,
    dropped_nodes_str: set,
) -> Tuple[Optional[str], Optional[str], Optional[List[str]], Optional[Tuple[int, int]]]:
    """
    If nodes_order_{tag}.txt exists, write pruned nodes_order.
    If cluster_matrix_{tag}.pt exists AND nodes_order exists, prune columns and write pruned matrix.
    Returns:
      out_nodes_order_path, out_matrix_path, pruned_nodes_order, pruned_matrix_shape
    """
    nodes_order_path = os.path.join(saved_dir, f"nodes_order_{tag}.txt")
    matrix_path = os.path.join(saved_dir, f"cluster_matrix_{tag}.pt")

    out_nodes_order_path = None
    out_matrix_path = None
    pruned_nodes_order = None
    pruned_shape = None

    if not os.path.exists(nodes_order_path):
        print(f"[INFO] {nodes_order_path} not found; skipping nodes_order/matrix pruning.")
        return None, None, None, None

    nodes_order = read_nodes_order(nodes_order_path)
    keep_mask = [node not in dropped_nodes_str for node in nodes_order]
    pruned_nodes_order = [n for n, keep in zip(nodes_order, keep_mask) if keep]

    out_nodes_order_path = os.path.join(saved_dir, f"nodes_order_{tag}{out_suffix}.txt")
    write_nodes_order(out_nodes_order_path, pruned_nodes_order)

    if not os.path.exists(matrix_path):
        print(f"[INFO] {matrix_path} not found; skipping cluster_matrix pruning.")
        return out_nodes_order_path, None, pruned_nodes_order, None

    if torch is None:
        raise RuntimeError("torch is required to prune cluster_matrix .pt. pip install torch")

    mat = load_cluster_matrix_pt(matrix_path)
    K, N = int(mat.shape[0]), int(mat.shape[1])
    if N != len(nodes_order):
        raise ValueError(
            f"cluster_matrix second dim ({N}) != nodes_order length ({len(nodes_order)}). "
            "Refuse to prune (misalignment risk)."
        )

    idx = torch.tensor([i for i, keep in enumerate(keep_mask) if keep], dtype=torch.long)
    pruned_mat = mat.index_select(dim=1, index=idx)

    out_matrix_path = os.path.join(saved_dir, f"cluster_matrix_{tag}{out_suffix}.pt")
    save_cluster_matrix_pt(pruned_mat, out_matrix_path)
    pruned_shape = (K, int(pruned_mat.shape[1]))

    return out_nodes_order_path, out_matrix_path, pruned_nodes_order, pruned_shape


def build_manifest(
    tag: str,
    num_clusters: int,
    num_nodes: int,
    num_edges: int,
    cluster_sizes: Dict[str, int],
    dropped_nodes_audit: str,
    notes: str,
    matrix_path: Optional[str],
    matrix_shape: Optional[List[int]],
) -> Dict[str, Any]:
    return {
        "tag": tag,
        "num_clusters": int(num_clusters),
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "labels_path": None,  # per your preference
        "matrix": {
            "format": "pt",
            "path": os.path.abspath(matrix_path) if matrix_path else None,
            "shape": matrix_shape if matrix_shape else None,
            "dtype": "bool",
        },
        "cluster_sizes": cluster_sizes,
        "dropped_nodes_audit": os.path.abspath(dropped_nodes_audit),
        "notes": notes,
    }


def assert_other_clusters_unchanged(
    before: Counter,
    after: Counter,
    target_cluster: int,
):
    """
    Enforce your invariant:
      clusters != target_cluster must keep the same counts
      target_cluster can only decrease (or same if nothing pruned)
    """
    # Build union keys so missing becomes 0
    keys = set(before.keys()) | set(after.keys())
    for k in keys:
        b = int(before.get(k, 0))
        a = int(after.get(k, 0))
        if k == target_cluster:
            if a > b:
                raise AssertionError(
                    f"[INVARIANT FAIL] target cluster {target_cluster} increased: before {b}, after {a}"
                )
        else:
            if a != b:
                raise AssertionError(
                    f"[INVARIANT FAIL] cluster {k} changed: before {b}, after {a}. "
                    f"You asked to prune ONLY cluster {target_cluster}."
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=DEFAULTS["tag"])
    ap.add_argument("--graph-pkl", default=DEFAULTS["graph_pkl"])
    ap.add_argument("--out-graph", default=DEFAULTS["out_graph"])
    ap.add_argument("--saved-dir", default=DEFAULTS["saved_dir"])
    ap.add_argument("--out-suffix", default=DEFAULTS["out_suffix"])

    ap.add_argument("--cluster-attr", default=DEFAULTS["cluster_attr"], help="graph node attribute for cluster id")
    ap.add_argument("--target-cluster", type=int, default=DEFAULTS["target_cluster"], help="ONLY prune nodes from this cluster id")

    ap.add_argument("--mode", choices=["isolates", "no_positive"], default=DEFAULTS["mode"])
    ap.add_argument("--weight-attr", default=DEFAULTS["weight_attr"])

    ap.add_argument("--no-artifact-prune", action="store_true", help="only prune graph + write audit/manifest; do NOT prune nodes_order/matrix")
    args = ap.parse_args()

    tag = args.tag
    saved_dir = args.saved_dir
    out_suffix = args.out_suffix

    print("[PATHS]")
    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("GRAPH_PKL    :", os.path.abspath(args.graph_pkl))
    print("OUT_GRAPH    :", os.path.abspath(args.out_graph))
    print("SAVED_DIR    :", os.path.abspath(saved_dir))
    print("MODE         :", args.mode)
    print("CLUSTER_ATTR :", args.cluster_attr)
    print("TARGET_CLS   :", args.target_cluster)
    print("WEIGHT_ATTR  :", args.weight_attr)
    print("OUT_SUFFIX   :", out_suffix)
    print("ARTIFACTS    :", "skip" if args.no_artifact_prune else "prune nodes_order/matrix if present")

    G = load_graph(args.graph_pkl)
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Loaded object is not a NetworkX graph: {type(G)}")

    # BEFORE sizes (graph truth)
    before_cnt = graph_cluster_sizes(G, args.cluster_attr)
    print("\n[BEFORE cluster_sizes (graph truth)]")
    for k in sorted(before_cnt.keys()):
        print(f"  cluster {k}: {before_cnt[k]}")

    # Decide nodes to drop (ONLY target cluster)
    dropped_nodes = nodes_to_drop_only_target_cluster(
        G=G,
        cluster_attr=args.cluster_attr,
        target_cluster=args.target_cluster,
        mode=args.mode,
        weight_attr=args.weight_attr,
    )
    dropped_nodes_str = {normalize_node_id(n) for n in dropped_nodes}
    print(f"\n[PRUNE] will drop {len(dropped_nodes)} nodes (ONLY from cluster {args.target_cluster})")

    # Prune graph
    Gp = G.copy()
    Gp.remove_nodes_from(dropped_nodes)
    save_graph(Gp, args.out_graph)

    # AFTER sizes (graph truth)
    after_cnt = graph_cluster_sizes(Gp, args.cluster_attr)
    print("\n[AFTER cluster_sizes (graph truth)]")
    for k in sorted(after_cnt.keys()):
        print(f"  cluster {k}: {after_cnt[k]}")

    # Enforce your invariant
    assert_other_clusters_unchanged(before_cnt, after_cnt, target_cluster=args.target_cluster)
    print("\n[OK] Invariant holds: clusters other than target are unchanged.")

    # Audit
    audit_path = os.path.join(saved_dir, f"dropped_nodes_{tag}{out_suffix}.json")
    os.makedirs(saved_dir, exist_ok=True)
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tag": tag,
                "cluster_attr": args.cluster_attr,
                "target_cluster_only": int(args.target_cluster),
                "mode": args.mode,
                "weight_attr": args.weight_attr if args.mode == "no_positive" else None,
                "dropped_count": int(len(dropped_nodes_str)),
                "dropped_nodes": sorted(list(dropped_nodes_str)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Optional artifact pruning (nodes_order + matrix)
    out_nodes_order_path = None
    out_matrix_path = None
    matrix_shape = None
    if not args.no_artifact_prune:
        out_nodes_order_path, out_matrix_path, _, pruned_shape = prune_nodes_order_and_matrix_optional(
            saved_dir=saved_dir,
            tag=tag,
            out_suffix=out_suffix,
            dropped_nodes_str=dropped_nodes_str,
        )
        if pruned_shape is not None:
            matrix_shape = [int(pruned_shape[0]), int(pruned_shape[1])]

    # Manifest cluster_sizes must reflect GRAPH TRUTH (your requirement)
    # num_clusters: use observed unique cluster ids in graph (can include gaps, but that's fine)
    num_clusters = len(after_cnt.keys())
    cluster_sizes_manifest = {str(k): int(v) for k, v in after_cnt.items()}

    note = (
        f"This run prunes ONLY nodes from cluster {args.target_cluster} based on graph node attribute '{args.cluster_attr}'. "
        f"Clusters other than {args.target_cluster} are guaranteed unchanged in size. "
        f"Prune mode='{args.mode}'."
    )

    manifest = build_manifest(
        tag=tag,
        num_clusters=num_clusters,
        num_nodes=Gp.number_of_nodes(),
        num_edges=Gp.number_of_edges(),
        cluster_sizes=cluster_sizes_manifest,
        dropped_nodes_audit=audit_path,
        notes=note,
        matrix_path=out_matrix_path,
        matrix_shape=matrix_shape,
    )

    manifest_path = os.path.join(saved_dir, f"manifest_{tag}{out_suffix}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n=== DONE ===")
    print(f"Pruned graph:        {os.path.abspath(args.out_graph)}")
    print(f"Audit:              {os.path.abspath(audit_path)}")
    print(f"Manifest:           {os.path.abspath(manifest_path)}")
    if out_nodes_order_path:
        print(f"Pruned nodes_order:  {os.path.abspath(out_nodes_order_path)}")
    if out_matrix_path:
        print(f"Pruned matrix:       {os.path.abspath(out_matrix_path)}")
    print(f"Dropped nodes:       {len(dropped_nodes_str)} (ONLY from cluster {args.target_cluster})")


if __name__ == "__main__":
    main()
