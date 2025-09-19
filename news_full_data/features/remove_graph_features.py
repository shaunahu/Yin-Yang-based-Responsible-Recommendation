#!/usr/bin/env python3
"""
remove_graph_features.py

Remove features (attributes) from a NetworkX graph and update metadata.

Examples
--------
# Remove one edge feature (in-place)
python remove_graph_features.py \
  --graph ../newsGraph/graph_with_edges.pkl \
  --meta  ../newsGraph/graph_with_edges_meta.json \
  --edge_features semantic_similarity

# Remove multiple edge features
python remove_graph_features.py \
  --graph ../newsGraph/graph_with_edges.pkl \
  --meta  ../newsGraph/graph_with_edges_meta.json \
  --edge_features semantic_similarity,cooccurrence

# Dry-run (show counts only)
python remove_graph_features.py \
  --graph ../newsGraph/graph_with_edges.pkl \
  --meta  ../newsGraph/graph_with_edges_meta.json \
  --edge_features semantic_similarity \
  --dry_run

# Save to a new file and keep original intact
python remove_graph_features.py \
  --graph ../newsGraph/graph_with_edges.pkl \
  --out   ../newsGraph/graph_without_semantic.pkl \
  --meta  ../newsGraph/graph_with_edges_meta.json \
  --edge_features semantic_similarity

# List the currently observed edge/node attributes
python remove_graph_features.py \
  --graph ../newsGraph/graph_with_edges.pkl \
  --list
"""
import os
import json
import pickle
import argparse
from collections import Counter

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


def load_graph(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def list_attributes(G):
    """Return (edge_attr_names, node_attr_names)."""
    edge_attrs = set()
    for _, _, data in G.edges(data=True):
        edge_attrs.update(data.keys())
    node_attrs = set()
    for _, data in G.nodes(data=True):
        node_attrs.update(data.keys())
    return sorted(edge_attrs), sorted(node_attrs)


def remove_edge_features(G, feature_names):
    """
    Remove each feature in feature_names from all edges.
    Returns a dict: {feature_name: count_removed}
    """
    stats = Counter()
    iterator = G.edges(data=True)
    if _HAS_TQDM:
        iterator = tqdm(iterator, total=G.number_of_edges(), desc="Removing edge features")

    for u, v, data in iterator:
        for f in feature_names:
            if f in data:
                del data[f]
                stats[f] += 1
    return dict(stats)


def remove_node_features(G, feature_names):
    """
    Remove each feature in feature_names from all nodes.
    Returns a dict: {feature_name: count_removed}
    """
    stats = Counter()
    iterator = G.nodes(data=True)
    if _HAS_TQDM:
        iterator = tqdm(iterator, total=G.number_of_nodes(), desc="Removing node features")

    for n, data in iterator:
        for f in feature_names:
            if f in data:
                del data[f]
                stats[f] += 1
    return dict(stats)


def update_meta(meta_path, graph_filename, G, removed_edge_stats, removed_node_stats):
    """
    Update the metadata JSON:
      - Remove names from meta['edge_features'] if present
      - Drop counts in meta['edge_feature_counts']
      - Drop per-feature model records in meta['models']
      - Add 'last_feature_removed' record
    """
    import datetime

    meta = {}
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    # Core
    meta["graph_file"] = os.path.basename(graph_filename)
    meta["nodes"] = int(G.number_of_nodes())
    meta["edges"] = int(G.number_of_edges())

    # Edge features
    if removed_edge_stats:
        # remove from list
        current_edge_feats = set(meta.get("edge_features", []))
        to_remove = set(removed_edge_stats.keys())
        remaining = current_edge_feats - to_remove
        meta["edge_features"] = sorted(remaining)

        # remove per-feature counts
        efc = meta.get("edge_feature_counts", {})
        for f in removed_edge_stats.keys():
            if f in efc:
                del efc[f]
        meta["edge_feature_counts"] = efc

        # remove models entries
        models = meta.get("models", {})
        for f in removed_edge_stats.keys():
            if f in models:
                del models[f]
        meta["models"] = models

    # Node features: (optional) you can track node-feature removals too if you maintain such metadata
    if removed_node_stats:
        meta.setdefault("node_feature_removals", {})
        for f, c in removed_node_stats.items():
            meta["node_feature_removals"][f] = int(c)

    # Log
    meta["last_feature_removed"] = {
        "edge_features": {k: int(v) for k, v in (removed_edge_stats or {}).items()},
        "node_features": {k: int(v) for k, v in (removed_node_stats or {}).items()},
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds")
    }

    if meta_path:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[INFO] Metadata updated -> {meta_path}")


def parse_args():
    HERE = os.path.dirname(os.path.abspath(__file__))
    default_graph = os.path.abspath(os.path.join(HERE, "..", "newsGraph", "graph_with_edges.pkl"))
    default_meta  = os.path.abspath(os.path.join(HERE, "..", "newsGraph", "graph_with_edges_meta.json"))

    ap = argparse.ArgumentParser(description="Remove features from a NetworkX graph and update metadata.")
    ap.add_argument("--graph", default=default_graph,
                    help="Path to input graph pickle (default: newsGraph/graph_with_edges.pkl)")
    ap.add_argument("--out", default=None,
                    help="Path to output graph pickle (default: overwrite input)")
    ap.add_argument("--meta", default=default_meta,
                    help="Path to metadata JSON (default: newsGraph/graph_with_edges_meta.json)")
    ap.add_argument("--edge_features", default="",
                    help="Comma-separated edge features to remove (e.g., 'semantic_similarity,cooccurrence')")
    ap.add_argument("--node_features", default="",
                    help="Comma-separated node features to remove (optional)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Show what would be removed, do not write files")
    ap.add_argument("--list", action="store_true",
                    help="List current edge/node attributes and exit")
    return ap.parse_args()


def main():
    args = parse_args()

    # Load graph
    G = load_graph(args.graph)
    print(f"[INFO] Loaded graph: nodes={G.number_of_nodes():,}, edges={G.number_of_edges():,}")

    if args.list:
        edge_attrs, node_attrs = list_attributes(G)
        print("[INFO] Observed edge attributes:", edge_attrs)
        print("[INFO] Observed node attributes:", node_attrs)
        return

    # Parse feature lists
    edge_feats = [s.strip() for s in args.edge_features.split(",") if s.strip()]
    node_feats = [s.strip() for s in args.node_features.split(",") if s.strip()]

    if not edge_feats and not node_feats:
        print("[WARN] No features specified. Use --edge_features and/or --node_features.")
        return

    # Remove edge features
    removed_edge_stats = {}
    if edge_feats:
        removed_edge_stats = remove_edge_features(G, edge_feats)
        for f, c in removed_edge_stats.items():
            print(f"[DONE] Removed edge feature '{f}' from {c:,} edges.")

    # Remove node features (optional)
    removed_node_stats = {}
    if node_feats:
        removed_node_stats = remove_node_features(G, node_feats)
        for f, c in removed_node_stats.items():
            print(f"[DONE] Removed node feature '{f}' from {c:,} nodes.")

    # Dry-run: do not save anything
    if args.dry_run:
        print("[DRY-RUN] No files written.")
        return

    # Decide output graph path
    out_path = args.out or args.graph
    # Save graph
    with open(out_path, "wb") as f:
        pickle.dump(G, f)
    print(f"[INFO] Saved graph -> {out_path}")

    # Update metadata JSON
    if args.meta:
        update_meta(
            meta_path=args.meta,
            graph_filename=os.path.basename(out_path),
            G=G,
            removed_edge_stats=removed_edge_stats,
            removed_node_stats=removed_node_stats
        )


if __name__ == "__main__":
    main()
