#!/usr/bin/env python3
# add_sentiment_to_graph_from_graph.py
# Sep 19,2025
import os
import json
import pickle
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from textblob import TextBlob


# ---------------------------- helpers ----------------------------

def load_graph(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph not found: {path}")
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G


def normalize_polarity_to_unit(x: float) -> float:
    # TextBlob polarity in [-1, 1] -> [0, 1]
    return (float(x) + 1.0) / 2.0


def per_item_sentiment(text: str) -> float:
    """Return normalized sentiment in [0,1] for a given text."""
    if not isinstance(text, str) or not text.strip():
        return 0.5  # neutral fallback
    p = TextBlob(text).sentiment.polarity  # [-1, 1]
    return normalize_polarity_to_unit(p)   # [0, 1]


def build_node_texts(
    G,
    items_csv: str,
    prefer_node_attrs: bool = True
) -> Tuple[List[str], List[str], List[str]]:
    """
    For each node (in G.nodes() order), produce (nodes, titles, abstracts).
    Prefer node attributes if present; otherwise look up in items_filtered.csv.
    """
    nodes = list(G.nodes())

    # Check attrs present on nodes
    has_title_attr = prefer_node_attrs and any("title" in G.nodes[n] for n in nodes)
    has_abs_attr   = prefer_node_attrs and any("abstract" in G.nodes[n] for n in nodes)

    id2row = {}
    titles_csv: List[str] = []
    abstracts_csv: List[str] = []

    if not (has_title_attr and has_abs_attr):
        if not os.path.exists(items_csv):
            raise FileNotFoundError(f"Items file not found: {items_csv}")
        df = pd.read_csv(items_csv, dtype=str)
        if "item_id" not in df.columns:
            raise ValueError(f"{items_csv} must contain 'item_id' column")
        id2row = {iid: idx for idx, iid in enumerate(df["item_id"].tolist())}
        titles_csv = df.get("title", pd.Series([""] * len(df))).fillna("").tolist()
        abstracts_csv = df.get("abstract", pd.Series([""] * len(df))).fillna("").tolist()

    titles: List[str] = []
    abstracts: List[str] = []
    missing = 0

    for n in nodes:
        t = ""
        a = ""
        if has_title_attr and "title" in G.nodes[n]:
            t = G.nodes[n].get("title") or ""
        if has_abs_attr and "abstract" in G.nodes[n]:
            a = G.nodes[n].get("abstract") or ""

        if (t == "" and a == "") and id2row:
            if n in id2row:
                r = id2row[n]
                t = titles_csv[r]
                a = abstracts_csv[r]
            else:
                missing += 1

        titles.append(t)
        abstracts.append(a)

    if missing > 0:
        print(f"[WARN] {missing} nodes not found in items_filtered.csv; filled with empty text.")
    return nodes, titles, abstracts


def compute_per_node_sentiment(titles: List[str], abstracts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-node sentiment for title and abstract, each in [0,1].
    """
    n = len(titles)
    title_sent = np.zeros(n, dtype=np.float32)
    abs_sent   = np.zeros(n, dtype=np.float32)

    for i in tqdm(range(n), desc="per-node sentiment (title & abstract)"):
        title_sent[i] = per_item_sentiment(titles[i])
        abs_sent[i]   = per_item_sentiment(abstracts[i])

    return title_sent, abs_sent


def write_sentiment_similarity_on_edges(
    G,
    nodes: List[str],
    title_sent: np.ndarray,
    abs_sent: np.ndarray,
    feature_name: str
) -> int:
    """
    For each existing edge (u,v), compute:
      sim = 1 - (|Δtitle| + |Δabstract|)/2
    and set as edge attribute `feature_name`.
    Returns number of edges updated.
    """
    node2idx: Dict[str, int] = {nid: i for i, nid in enumerate(nodes)}
    updated = 0

    edges = list(G.edges())
    print(f"[INFO] Updating '{feature_name}' for {len(edges):,} existing edges...")
    for u, v in tqdm(edges, desc="sentiment similarity to edges"):
        iu = node2idx.get(u)
        iv = node2idx.get(v)
        if iu is None or iv is None:
            continue
        d_title = abs(float(title_sent[iu]) - float(title_sent[iv]))
        d_abs   = abs(float(abs_sent[iu])   - float(abs_sent[iv]))
        sim = 1.0 - (d_title + d_abs) / 2.0
        # clip to [0,1] for safety
        if sim < 0.0: sim = 0.0
        if sim > 1.0: sim = 1.0
        G[u][v][feature_name] = float(sim)
        updated += 1
    return updated


def update_meta(meta_path: str, graph_filename: str, G, feature_name: str, updated_count: int):
    """Append/merge feature info into graph meta JSON."""
    import datetime

    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    # Core fields
    meta.setdefault("graph_file", graph_filename)
    meta["nodes"] = int(G.number_of_nodes())
    meta["edges"] = int(G.number_of_edges())

    # Edge features list
    edge_feats = meta.get("edge_features", [])
    if feature_name not in edge_feats:
        edge_feats.append(feature_name)
    meta["edge_features"] = edge_feats

    # Per-feature counts
    efc = meta.get("edge_feature_counts", {})
    efc[feature_name] = int(updated_count)
    meta["edge_feature_counts"] = efc

    # Timestamp
    meta["last_updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    meta.setdefault("notes", {})
    meta["notes"][feature_name] = "sentiment_similarity = 1 - (|Δtitle| + |Δabstract|)/2, per TextBlob polarity normalized to [0,1]."

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Metadata updated -> {meta_path}")


# ---------------------------- main ----------------------------

def parse_args():
    HERE = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.abspath(os.path.join(HERE, "..", "data_news", "items_filtered.csv"))
    default_graph_dir = os.path.abspath(os.path.join(HERE, "..", "newsGraph"))

    ap = argparse.ArgumentParser(description="Add sentiment similarity to existing edges in a NetworkX graph.")
    ap.add_argument("--items_csv", default=default_data, help="Path to items_filtered.csv")
    ap.add_argument("--graph_in",  default=os.path.join(default_graph_dir, "graph.pkl"),
                    help="Input graph (with nodes+edges). Default: newsGraph/graph.pkl")
    ap.add_argument("--graph_out", default=os.path.join(default_graph_dir, "graph.pkl"),
                    help="Output graph path (default overwrites input)")
    ap.add_argument("--meta_out",  default=os.path.join(default_graph_dir, "graph_with_edges_meta.json"),
                    help="Metadata JSON path to update")
    ap.add_argument("--feature_name", default="sentiment_similarity",
                    help="Edge attribute name to write (default: sentiment_similarity)")
    ap.add_argument("--prefer_node_attrs", action="store_true",
                    help="Prefer existing node attrs (title/abstract) over CSV when available")
    return ap.parse_args()


def main():
    args = parse_args()

    # 1) Load graph
    G = load_graph(args.graph_in)
    print(f"[INFO] Loaded graph: nodes={G.number_of_nodes():,}, edges={G.number_of_edges():,}")

    # 2) Gather node texts
    nodes, titles, abstracts = build_node_texts(G, args.items_csv, prefer_node_attrs=args.prefer_node_attrs)

    # 3) Compute per-node sentiments
    title_sent, abs_sent = compute_per_node_sentiment(titles, abstracts)

    # 4) Write similarity to existing edges
    updated = write_sentiment_similarity_on_edges(G, nodes, title_sent, abs_sent, args.feature_name)

    # 5) Save graph
    with open(args.graph_out, "wb") as f:
        pickle.dump(G, f)
    print(f"[DONE] Wrote '{args.feature_name}' on {updated:,} edges.")
    print(f"[INFO] Saved graph -> {args.graph_out}")

    # 6) Update metadata
    update_meta(
        meta_path=args.meta_out,
        graph_filename=os.path.basename(args.graph_out),
        G=G,
        feature_name=args.feature_name,
        updated_count=updated
    )


if __name__ == "__main__":
    main()
