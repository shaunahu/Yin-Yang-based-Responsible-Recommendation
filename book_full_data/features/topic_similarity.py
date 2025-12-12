#!/usr/bin/env python3
# topic_similarity.py
# Insert category similarity into booksGraph/graph_with_edges.pkl
# BOOK dataset only
# Dec 12, 2025

import os
import json
import pickle
import argparse
import pandas as pd
from tqdm import tqdm


# --------------------
# Load graph
# --------------------
def load_graph(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------
# Load book topics
# --------------------
def load_topics(items_tsv: str) -> dict:
    df = pd.read_csv(items_tsv, dtype=str, sep="\t")

    required = {"bookid", "cat"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {items_tsv}")

    return {
        bid: cat.strip().lower() if isinstance(cat, str) else ""
        for bid, cat in zip(df["bookid"], df["cat"])
    }


# --------------------
# Load topic similarity matrix
# --------------------
def load_topic_similarity(path: str):
    sim = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            c1, c2, v = parts
            try:
                v = float(v)
            except ValueError:
                continue
            sim[(c1.lower(), c2.lower())] = v
            sim[(c2.lower(), c1.lower())] = v
    return sim


# --------------------
# Annotate edges
# --------------------
def write_topic_similarity(G, node_topics, topic_sim, feature_name):
    updated = 0
    print(f"[INFO] Computing {feature_name} for {G.number_of_edges():,} edges")

    for u, v in tqdm(G.edges(), desc=f"Adding {feature_name}"):
        t1 = node_topics.get(str(u), "")
        t2 = node_topics.get(str(v), "")

        if not t1 or not t2:
            sim = 0.0
        elif t1 == t2:
            sim = 1.0
        else:
            sim = topic_sim.get((t1, t2), 0.0)

        G[u][v][feature_name] = float(sim)
        updated += 1

    return updated


# --------------------
# Metadata
# --------------------
def update_meta(meta_path, graph_filename, G, feature_name, updated):
    import datetime

    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            pass

    meta["graph_file"] = graph_filename
    meta["nodes"] = G.number_of_nodes()
    meta["edges"] = G.number_of_edges()

    feats = meta.get("edge_features", [])
    if feature_name not in feats:
        feats.append(feature_name)
    meta["edge_features"] = feats

    counts = meta.get("edge_feature_counts", {})
    counts[feature_name] = updated
    meta["edge_feature_counts"] = counts

    meta["last_updated"] = datetime.datetime.now().isoformat(timespec="seconds")

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


# --------------------
# CLI
# --------------------
def parse_args():
    HERE = os.path.dirname(os.path.abspath(__file__))
    GRAPH_DIR = os.path.join(HERE, "..", "booksGraph")

    return argparse.Namespace(
        items_tsv=os.path.join(HERE, "..", "data_book", "items_filtered.tsv"),
        topic_sim=os.path.join(HERE, "..", "data_book", "book-similarity-normalized.txt"),
        graph_in=os.path.join(GRAPH_DIR, "graph_with_edges.pkl"),
        graph_out=os.path.join(GRAPH_DIR, "graph_with_edges.pkl"),
        meta_out=os.path.join(GRAPH_DIR, "graph_with_edges_meta.json"),
        feature_name="topic_similarity",
    )


# --------------------
# MAIN
# --------------------
def main():
    args = parse_args()

    # Fallback if edges not created yet
    if not os.path.exists(args.graph_in):
        fallback = args.graph_in.replace("graph_with_edges.pkl", "graph.pkl")
        print(f"[WARN] graph_with_edges.pkl not found, falling back to {fallback}")
        graph_path = fallback
    else:
        graph_path = args.graph_in

    G = load_graph(graph_path)
    print(f"[INFO] Loaded graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    node_topics = load_topics(args.items_tsv)
    topic_sim = load_topic_similarity(args.topic_sim)

    updated = write_topic_similarity(G, node_topics, topic_sim, args.feature_name)

    with open(args.graph_out, "wb") as f:
        pickle.dump(G, f)

    print(f"[DONE] Wrote '{args.feature_name}' on {updated:,} edges")
    print(f"[INFO] Saved graph -> {args.graph_out}")

    update_meta(args.meta_out, os.path.basename(args.graph_out), G, args.feature_name, updated)


if __name__ == "__main__":
    main()
