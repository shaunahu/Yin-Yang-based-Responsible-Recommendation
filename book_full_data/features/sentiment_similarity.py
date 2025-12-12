#!/usr/bin/env python3
# sentiment_similarity.py
# Insert sentiment-based similarity (from abstract + summary)
# into an existing NetworkX graph_with_edges.pkl for the BOOK dataset only.
# Dec 12, 2025

import os
import json
import pickle
import argparse
import pandas as pd
from tqdm import tqdm

from textblob import TextBlob  # pip install textblob


# --------------------
# Load graph
# --------------------
def load_graph(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------
# Load book texts
# --------------------
def load_texts(items_tsv: str):
    """
    Load abstract + summary text for BOOK dataset.
    Returns:
        ids   : list of bookid strings
        texts : list of combined abstract + summary strings
    """

    if not os.path.exists(items_tsv):
        raise FileNotFoundError(f"Items TSV not found: {items_tsv}")

    df = pd.read_csv(items_tsv, dtype=str, sep="\t")

    required = {"bookid", "abstract", "summary"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{items_tsv} must contain columns {required}, missing: {missing}"
        )

    ids = df["bookid"].astype(str).tolist()
    abs_col = df["abstract"].fillna("").astype(str)
    sum_col = df["summary"].fillna("").astype(str)

    texts = (abs_col + " " + sum_col).str.strip().tolist()
    return ids, texts


# --------------------
# Compute per-book sentiment
# --------------------
def compute_sentiment(ids, texts):
    """
    Compute sentiment polarity using TextBlob.
    Returns dict: bookid -> polarity [-1, 1]
    """
    id_to_polarity = {}
    print("[INFO] Computing sentiment polarity (TextBlob) for each book...")

    for bid, txt in tqdm(list(zip(ids, texts)), desc="Sentiment per book"):
        if not txt.strip():
            polarity = 0.0
        else:
            polarity = TextBlob(txt).sentiment.polarity
        id_to_polarity[bid] = float(polarity)

    return id_to_polarity


# --------------------
# Write edge similarity
# --------------------
def write_sentiment_similarity(G, id_to_polarity, feature_name: str) -> int:
    updated = 0
    missing = 0

    total_edges = G.number_of_edges()
    print(f"[INFO] Computing {feature_name} for {total_edges:,} edges...")

    for u, v in tqdm(G.edges(), desc=f"Adding {feature_name}"):
        p1 = id_to_polarity.get(str(u))
        p2 = id_to_polarity.get(str(v))

        if p1 is None or p2 is None:
            sim = 0.0
            missing += 1
        else:
            diff = abs(p1 - p2)
            sim = 1.0 - (diff / 2.0)  # map [-1, 1] -> [0, 1]

        G[u][v][feature_name] = float(sim)
        updated += 1

    print(f"[INFO] Finished: updated {updated:,} edges, missing sentiment={missing:,}")
    return updated


# --------------------
# Update metadata
# --------------------
def update_meta(meta_path, graph_filename, G, feature_name, updated_count):
    import datetime

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    else:
        meta = {}

    meta.setdefault("graph_file", graph_filename)
    meta["nodes"] = G.number_of_nodes()
    meta["edges"] = G.number_of_edges()

    feats = meta.get("edge_features", [])
    if feature_name not in feats:
        feats.append(feature_name)
    meta["edge_features"] = feats

    efc = meta.get("edge_feature_counts", {})
    efc[feature_name] = int(updated_count)
    meta["edge_feature_counts"] = efc

    meta.setdefault("notes", {})
    meta["notes"][feature_name] = (
        "Sentiment similarity from TextBlob polarity over abstract+summary (books dataset)."
    )

    meta["last_updated"] = datetime.datetime.now().isoformat(timespec="seconds")

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Metadata updated -> {meta_path}")


# --------------------
# CLI arguments
# --------------------
def parse_args():
    HERE = os.path.dirname(os.path.abspath(__file__))

    default_items = os.path.join(HERE, "..", "data_book", "items_filtered.tsv")
    default_graph_dir = os.path.join(HERE, "..", "booksGraph")

    # Default: operate directly on graph_with_edges.pkl (cooccurrence already added).
    default_graph_in = os.path.join(default_graph_dir, "graph_with_edges.pkl")
    default_meta = os.path.join(default_graph_dir, "graph_with_edges_meta.json")

    ap = argparse.ArgumentParser(
        description="Add sentiment-based similarity to booksGraph/graph_with_edges.pkl (books dataset only)."
    )

    ap.add_argument("--items_tsv", default=default_items)

    ap.add_argument(
        "--graph_in",
        default=default_graph_in,
        help="Input graph (default: booksGraph/graph_with_edges.pkl).",
    )
    ap.add_argument(
        "--graph_out",
        default=default_graph_in,
        help="Output graph (default: overwrite booksGraph/graph_with_edges.pkl).",
    )
    ap.add_argument(
        "--meta_out",
        default=default_meta,
        help="Metadata JSON path (default: booksGraph/graph_with_edges_meta.json).",
    )
    ap.add_argument(
        "--feature_name",
        default="sentiment_similarity",
        help="Edge attribute name (default: sentiment_similarity).",
    )

    return ap.parse_args()


# --------------------
# MAIN
# --------------------
def main():
    args = parse_args()

    graph_in = args.graph_in
    # If graph_with_edges.pkl doesn't exist yet, fall back to graph.pkl once
    if not os.path.exists(graph_in):
        fallback = os.path.join(os.path.dirname(graph_in), "graph.pkl")
        print(f"[WARN] {graph_in} not found, falling back to {fallback}")
        graph_in = fallback

    # Load graph (ideally graph_with_edges.pkl with cooccurrence edges)
    G = load_graph(graph_in)
    print(f"[INFO] Loaded graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Load book texts
    ids, texts = load_texts(args.items_tsv)
    print(f"[INFO] Loaded {len(ids):,} items with abstract+summary.")

    # Compute polarity
    id_to_polarity = compute_sentiment(ids, texts)

    # Add sentiment_similarity to edges
    updated = write_sentiment_similarity(G, id_to_polarity, args.feature_name)

    # Save graph (graph_with_edges.pkl by default)
    with open(args.graph_out, "wb") as f:
        pickle.dump(G, f)

    print(f"[DONE] Wrote '{args.feature_name}' on {updated:,} edges")
    print(f"[INFO] Saved graph -> {args.graph_out}")

    # Update metadata JSON
    update_meta(args.meta_out, os.path.basename(args.graph_out), G, args.feature_name, updated)


if __name__ == "__main__":
    main()
