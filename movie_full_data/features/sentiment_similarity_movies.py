#!/usr/bin/env python3
# sentiment_similarity.py
# Insert sentiment-based similarity (from abstract + summary)
# into an existing NetworkX graph_with_edges.pkl for the MOVIE dataset only.
# Dec 11, 2025

import os
import json
import pickle
import argparse
import pandas as pd
from tqdm import tqdm

from textblob import TextBlob  # pip install textblob


def load_graph(path: str):
    """Load a pickled NetworkX graph."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph not found: {path}")
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G


def load_texts(items_tsv: str):
    """
    Load abstract + summary text for the MOVIE dataset.
    Returns:
        ids   : list of movieid strings
        texts : list of combined "abstract summary" strings (lowercased)
    """

    if not os.path.exists(items_tsv):
        raise FileNotFoundError(f"Items TSV not found: {items_tsv}")

    df = pd.read_csv(items_tsv, dtype=str, sep="\t")

    required = {"movieid", "abstract", "summary"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{items_tsv} must contain columns {required}, missing: {missing}"
        )

    ids = df["movieid"].astype(str).tolist()

    abs_col = df["abstract"].fillna("").astype(str)
    sum_col = df["summary"].fillna("").astype(str)

    texts = (abs_col + " " + sum_col).str.strip().tolist()
    return ids, texts


def compute_sentiment(ids, texts):
    """
    Compute sentiment polarity for each movie text using TextBlob.

    Returns:
        id_to_polarity: dict movieid -> polarity in [-1, 1]
    """
    id_to_polarity = {}
    print("[INFO] Computing sentiment polarity (TextBlob) for each movie...")

    for mid, txt in tqdm(list(zip(ids, texts)), desc="Sentiment per movie"):
        if not txt.strip():
            polarity = 0.0
        else:
            polarity = TextBlob(txt).sentiment.polarity  # [-1, 1]
        id_to_polarity[mid] = float(polarity)

    return id_to_polarity


def write_sentiment_similarity(G, id_to_polarity, feature_name: str) -> int:
    """
    For each existing edge, compute sentiment similarity in [0,1] from polarity.

    We define:
        p1, p2 in [-1, 1]
        diff = |p1 - p2|  in [0, 2]
        sim = 1 - diff / 2  in [0, 1]

    and store sim as edge attribute feature_name.
    """
    updated = 0
    missing = 0

    total_edges = G.number_of_edges()
    print(f"[INFO] Computing '{feature_name}' for {total_edges:,} edges...")

    for u, v in tqdm(G.edges(), desc=f"Adding {feature_name}"):
        p1 = id_to_polarity.get(str(u), None)
        p2 = id_to_polarity.get(str(v), None)

        if p1 is None or p2 is None:
            sim = 0.0
            missing += 1
        else:
            diff = abs(p1 - p2)          # [0, 2]
            sim = 1.0 - (diff / 2.0)     # [0, 1]

        G[u][v][feature_name] = float(sim)
        updated += 1

    print(f"[INFO] Finished '{feature_name}': updated {updated:,} edges.")
    print(f"[INFO] Edges with at least one missing sentiment: {missing:,}")
    return updated


def update_meta(meta_path: str, graph_filename: str, G, feature_name: str, updated_count: int):
    """Update graph metadata JSON."""
    import datetime

    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    meta.setdefault("graph_file", graph_filename)
    meta["nodes"] = int(G.number_of_nodes())
    meta["edges"] = int(G.number_of_edges())

    # Track edge features
    edge_feats = meta.get("edge_features", [])
    if feature_name not in edge_feats:
        edge_feats.append(feature_name)
    meta["edge_features"] = edge_feats

    efc = meta.get("edge_feature_counts", {})
    efc[feature_name] = int(updated_count)
    meta["edge_feature_counts"] = efc

    meta["last_updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    meta.setdefault("notes", {})
    meta["notes"][feature_name] = (
        "Sentiment similarity from TextBlob polarity over abstract+summary (movies dataset)."
    )

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Metadata updated -> {meta_path}")


def parse_args():
    HERE = os.path.dirname(os.path.abspath(__file__))

    default_items = os.path.abspath(os.path.join(HERE, "..", "data_movie", "items_filtered.tsv"))
    default_graph_dir = os.path.abspath(os.path.join(HERE, "..", "moviesGraph"))

    ap = argparse.ArgumentParser(
        description="Add sentiment-based similarity to graph_with_edges.pkl (movies dataset only)."
    )

    ap.add_argument(
        "--items_tsv",
        default=default_items,
        help="Path to items_filtered.tsv (movies dataset).",
    )
    ap.add_argument(
        "--graph_in",
        default=os.path.join(default_graph_dir, "graph_with_edges.pkl"),
        help="Input graph (default: moviesGraph/graph_with_edges.pkl).",
    )
    ap.add_argument(
        "--graph_out",
        default=os.path.join(default_graph_dir, "graph_with_edges.pkl"),
        help="Output graph (default: overwrite graph_with_edges.pkl).",
    )
    ap.add_argument(
        "--meta_out",
        default=os.path.join(default_graph_dir, "graph_with_edges_meta.json"),
        help="Metadata JSON path.",
    )
    ap.add_argument(
        "--feature_name",
        default="sentiment_similarity",
        help="Edge attribute name (default: sentiment_similarity).",
    )

    return ap.parse_args()


def main():
    args = parse_args()

    # 1) Load graph
    G = load_graph(args.graph_in)
    print(f"[INFO] Loaded graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # 2) Load texts
    ids, texts = load_texts(args.items_tsv)
    print(f"[INFO] Loaded {len(ids):,} items with abstract+summary text.")

    # 3) Compute per-movie sentiment polarity
    id_to_polarity = compute_sentiment(ids, texts)

    # 4) Write sentiment similarity on existing edges
    updated = write_sentiment_similarity(G, id_to_polarity, args.feature_name)

    # 5) Save graph_with_edges.pkl
    with open(args.graph_out, "wb") as f:
        pickle.dump(G, f)

    print(f"[DONE] Wrote '{args.feature_name}' on {updated:,} edges.")
    print(f"[INFO] Saved graph -> {args.graph_out}")

    # 6) Update metadata JSON
    update_meta(
        args.meta_out,
        os.path.basename(args.graph_out),
        G,
        args.feature_name,
        updated,
    )


if __name__ == "__main__":
    main()
