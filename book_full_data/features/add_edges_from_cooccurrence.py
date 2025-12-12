#!/usr/bin/env python3
# add_edges_from_cooccurrence_books.py
# Add co-occurrence edges to the BOOKS graph from impression_cooccurrence_books.csv
# Dec 12, 2025

import os
import argparse
import pandas as pd
from tqdm import tqdm
import pickle
import json


def main(min_count: int, overwrite: bool):
    # Resolve paths
    HERE = os.path.dirname(os.path.abspath(__file__))
    GRAPH_DIR = os.path.abspath(os.path.join(HERE, "..", "booksGraph"))
    EDGE_DIR  = os.path.abspath(os.path.join(HERE, "..", "edge_data"))

    GRAPH_IN   = os.path.join(GRAPH_DIR, "graph.pkl")
    COOC_CSV   = os.path.join(EDGE_DIR, "impression_cooccurrence_books.csv")
    GRAPH_OUT  = os.path.join(GRAPH_DIR, "graph_with_edges.pkl")
    META_OUT   = os.path.join(GRAPH_DIR, "graph_with_edges_meta.json")

    # Overwrite option
    if overwrite:
        GRAPH_OUT = GRAPH_IN  # replace original graph
        META_OUT  = os.path.join(GRAPH_DIR, "graph_with_edges_meta.json")

    # ---- load graph ----
    if not os.path.exists(GRAPH_IN):
        raise FileNotFoundError(f"Graph not found: {GRAPH_IN}")
    with open(GRAPH_IN, "rb") as f:
        G = pickle.load(f)

    print(f"[INFO] Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ---- load co-occurrence CSV ----
    if not os.path.exists(COOC_CSV):
        raise FileNotFoundError(f"Cooccurrence CSV not found: {COOC_CSV}")

    df = pd.read_csv(
        COOC_CSV,
        dtype={"bookid_a": str, "bookid_b": str, "cooccurrence": int}
    )

    required = {"bookid_a", "bookid_b", "cooccurrence"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{COOC_CSV} must contain columns {sorted(required)}, found {list(df.columns)}"
        )

    # Optional filter
    if min_count > 1:
        before = len(df)
        df = df[df["cooccurrence"] >= min_count].copy()
        print(f"[INFO] Filtered by min_count={min_count}: {before} → {len(df)} rows")

    # ---- add edges ----
    nodes_set = set(G.nodes)
    added, updated, skipped = 0, 0, 0

    print("[INFO] Adding edges with 'cooccurrence' attribute (books dataset) ...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding edges"):
        u, v, c = row["bookid_a"], row["bookid_b"], int(row["cooccurrence"])

        if u not in nodes_set or v not in nodes_set:
            skipped += 1
            continue

        if G.has_edge(u, v):
            G[u][v]["cooccurrence"] = c
            updated += 1
        else:
            G.add_edge(u, v, cooccurrence=c)
            added += 1

    # ---- save graph ----
    with open(GRAPH_OUT, "wb") as f:
        pickle.dump(G, f)

    total_edges = G.number_of_edges()
    print(f"[DONE] Saved -> {GRAPH_OUT}")
    print(f"       Nodes: {G.number_of_nodes()} | Edges: {total_edges}")
    print(f"       Added: {added}, Updated: {updated}, Skipped (missing nodes): {skipped}")

    # ---- metadata ----
    meta = {
        "graph_file": os.path.basename(GRAPH_OUT),
        "nodes": G.number_of_nodes(),
        "edges": total_edges,
        "added_edges": added,
        "updated_edges": updated,
        "skipped_missing": skipped,
        "edge_features": ["cooccurrence"],
        "min_count": min_count,
        "note": (
            "Edge attribute 'cooccurrence' = number of times two books "
            "co-occurred in impression lists (books dataset)."
        ),
    }

    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Metadata saved -> {META_OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Add edges to BOOKS graph from impression_cooccurrence_books.csv"
    )
    ap.add_argument(
        "--min_count",
        type=int,
        default=1,
        help="Only add pairs with cooccurrence >= this value (default: 1)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite booksGraph/graph.pkl instead of creating graph_with_edges.pkl",
    )
    args = ap.parse_args()
    main(min_count=args.min_count, overwrite=args.overwrite)
