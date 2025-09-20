#!/usr/bin/env python3
# add_topic_to_graph_from_graph.py
# Insert topic similarity into an existing NetworkX graph.pkl
# Sep 19,2025

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_graph(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph not found: {path}")
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G


def load_topics(items_csv: str) -> dict:
    """Return dict item_id -> topic (lowercased string)."""
    df = pd.read_csv(items_csv, dtype=str)
    if "item_id" not in df.columns or "topic" not in df.columns:
        raise ValueError(f"{items_csv} must contain item_id and topic columns.")
    topics = df["topic"].fillna("").str.strip().str.lower().tolist()
    return {iid: topics[i] for i, iid in enumerate(df["item_id"].tolist())}


def load_topic_similarity(topic_sim_txt: str):
    """Load topic similarity file into dict of (t1,t2)->sim (symmetric)."""
    sim_dict = {}
    with open(topic_sim_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            t1, t2, val = parts[0].lower(), parts[1].lower(), parts[2]
            try:
                sim = float(val)
            except ValueError:
                continue
            sim_dict[(t1, t2)] = sim
            sim_dict[(t2, t1)] = sim
    return sim_dict


def write_topic_similarity(G, node_topics: dict, topic_sim: dict, feature_name: str) -> int:
    """Compute topic similarity for each existing edge and store as edge attr."""
    updated = 0
    for u, v in tqdm(G.edges(), desc=f"Adding {feature_name}"):
        t1 = node_topics.get(u, "")
        t2 = node_topics.get(v, "")
        if not t1 or not t2:
            sim = 0.0
        elif t1 == t2:
            sim = 1.0
        else:
            sim = topic_sim.get((t1, t2), 0.0)  # default to 0 if missing
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

    meta.setdefault("graph_file", graph_filename)
    meta["nodes"] = int(G.number_of_nodes())
    meta["edges"] = int(G.number_of_edges())

    edge_feats = meta.get("edge_features", [])
    if feature_name not in edge_feats:
        edge_feats.append(feature_name)
    meta["edge_features"] = edge_feats

    efc = meta.get("edge_feature_counts", {})
    efc[feature_name] = int(updated_count)
    meta["edge_feature_counts"] = efc

    meta["last_updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    meta.setdefault("notes", {})
    meta["notes"][feature_name] = "Topic similarity from news_topic_similarity_normalized.txt."

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Metadata updated -> {meta_path}")


def parse_args():
    HERE = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.abspath(os.path.join(HERE, "..", "data_news", "items_filtered.csv"))
    default_edge = os.path.abspath(os.path.join(HERE, "..", "edge_data", "news_topic_similarity_normalized.txt"))
    default_graph_dir = os.path.abspath(os.path.join(HERE, "..", "newsGraph"))

    ap = argparse.ArgumentParser(description="Add topic similarity to existing edges in a NetworkX graph.")
    ap.add_argument("--items_csv", default=default_data, help="Path to items_filtered.csv")
    ap.add_argument("--topic_sim", default=default_edge, help="Path to news_topic_similarity_normalized.txt")
    ap.add_argument("--graph_in", default=os.path.join(default_graph_dir, "graph.pkl"),
                    help="Input graph (default: newsGraph/graph.pkl)")
    ap.add_argument("--graph_out", default=os.path.join(default_graph_dir, "graph.pkl"),
                    help="Output graph (default: overwrite input)")
    ap.add_argument("--meta_out", default=os.path.join(default_graph_dir, "graph_with_edges_meta.json"),
                    help="Path to metadata JSON")
    ap.add_argument("--feature_name", default="topic_similarity", help="Edge attribute name (default: topic_similarity)")
    return ap.parse_args()


def main():
    args = parse_args()

    # Load resources
    G = load_graph(args.graph_in)
    print(f"[INFO] Loaded graph: nodes={G.number_of_nodes():,}, edges={G.number_of_edges():,}")
    node_topics = load_topics(args.items_csv)
    topic_sim = load_topic_similarity(args.topic_sim)

    # Update edges
    updated = write_topic_similarity(G, node_topics, topic_sim, args.feature_name)

    # Save graph
    with open(args.graph_out, "wb") as f:
        pickle.dump(G, f)
    print(f"[DONE] Wrote '{args.feature_name}' on {updated:,} edges.")
    print(f"[INFO] Saved graph -> {args.graph_out}")

    # Update metadata
    update_meta(args.meta_out, os.path.basename(args.graph_out), G, args.feature_name, updated)


if __name__ == "__main__":
    main()
