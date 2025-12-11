#!/usr/bin/env python3
# topic_similarity.py
# Insert category similarity (from movie-similarity-normalized.txt)
# into an existing NetworkX graph_with_edges.pkl for the MOVIE dataset only.
# Dec 11, 2025

import os
import json
import pickle
import argparse
import pandas as pd
from tqdm import tqdm


def load_graph(path: str):
    """Load a pickled NetworkX graph."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph not found: {path}")
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G


def load_topics(items_tsv: str) -> dict:
    """
    Return dict: movieid -> category (cat1, lowercased string).

    Expects TSV with at least:
        movieid, cat1
    """
    if not os.path.exists(items_tsv):
        raise FileNotFoundError(f"Items TSV not found: {items_tsv}")

    df = pd.read_csv(items_tsv, dtype=str, sep="\t")

    required = {"movieid", "cat1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{items_tsv} must contain columns {required}, missing: {missing}"
        )

    cats = df["cat1"].fillna("").str.strip().str.lower().tolist()
    ids = df["movieid"].astype(str).tolist()
    return {mid: cats[i] for i, mid in enumerate(ids)}


def load_topic_similarity(topic_sim_txt: str):
    """
    Load movie category similarity file into dict of (c1, c2) -> sim (symmetric).

    Expects each non-empty line to be:
        cat1  cat2  similarity_value
    """
    if not os.path.exists(topic_sim_txt):
        raise FileNotFoundError(f"Topic similarity file not found: {topic_sim_txt}")

    sim_dict = {}
    with open(topic_sim_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            c1, c2, val = parts[0].lower(), parts[1].lower(), parts[2]
            try:
                sim = float(val)
            except ValueError:
                continue
            sim_dict[(c1, c2)] = sim
            sim_dict[(c2, c1)] = sim
    return sim_dict


def write_topic_similarity(G, node_topics: dict, topic_sim: dict, feature_name: str) -> int:
    """
    Compute category similarity for each EXISTING edge and store as edge attr.

    IMPORTANT: This does NOT create edges; it only annotates edges already in G.
    """
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
    """Append/merge feature info into a graph meta JSON (movie-specific)."""
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
    meta["notes"][feature_name] = (
        "Category similarity from movie-similarity-normalized.txt (movies dataset)."
    )

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Metadata updated -> {meta_path}")


def parse_args():
    HERE = os.path.dirname(os.path.abspath(__file__))

    # MOVIE-specific defaults
    default_items = os.path.abspath(
        os.path.join(HERE, "..", "data_movie", "items_filtered.tsv")
    )
    default_sim = os.path.abspath(
        os.path.join(HERE, "..", "data_movie", "movie-similarity-normalized.txt")
    )
    default_graph_dir = os.path.abspath(
        os.path.join(HERE, "..", "moviesGraph")
    )

    ap = argparse.ArgumentParser(
        description=(
            "Add movie category similarity to existing edges in a NetworkX graph "
            "(movies dataset only)."
        )
    )
    ap.add_argument(
        "--items_tsv",
        default=default_items,
        help="Path to data_movie/items_filtered.tsv (movies dataset).",
    )
    ap.add_argument(
        "--topic_sim",
        default=default_sim,
        help="Path to data_movie/movie-similarity-normalized.txt.",
    )
    ap.add_argument(
        "--graph_in",
        default=os.path.join(default_graph_dir, "graph_with_edges.pkl"),
        help="Input graph (default: moviesGraph/graph_with_edges.pkl).",
    )
    ap.add_argument(
        "--graph_out",
        default=os.path.join(default_graph_dir, "graph_with_edges.pkl"),
        help="Output graph (default: overwrite moviesGraph/graph_with_edges.pkl).",
    )
    ap.add_argument(
        "--meta_out",
        default=os.path.join(default_graph_dir, "graph_with_edges_meta.json"),
        help="Path to metadata JSON.",
    )
    ap.add_argument(
        "--feature_name",
        default="topic_similarity",
        help="Edge attribute name (default: topic_similarity).",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    # Load resources
    G = load_graph(args.graph_in)
    print(
        f"[INFO] Loaded graph: nodes={G.number_of_nodes():,}, "
        f"edges={G.number_of_edges():,}"
    )

    node_topics = load_topics(args.items_tsv)
    topic_sim = load_topic_similarity(args.topic_sim)

    # Update edges
    updated = write_topic_similarity(G, node_topics, topic_sim, args.feature_name)

    # Save graph (graph_with_edges.pkl by default)
    with open(args.graph_out, "wb") as f:
        pickle.dump(G, f)
    print(f"[DONE] Wrote '{args.feature_name}' on {updated:,} edges.")
    print(f"[INFO] Saved graph -> {args.graph_out}")

    # Update metadata
    update_meta(
        args.meta_out,
        os.path.basename(args.graph_out),
        G,
        args.feature_name,
        updated,
    )


if __name__ == "__main__":
    main()
