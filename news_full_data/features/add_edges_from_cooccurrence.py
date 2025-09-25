# add_edges_from_cooccurrence.py
# Sep 19,2025
import os
import argparse
import pandas as pd
import networkx as nx
from tqdm import tqdm
import pickle
import json

def main(min_count: int, overwrite: bool):
    # Resolve paths
    HERE = os.path.dirname(os.path.abspath(__file__))
    GRAPH_DIR = os.path.abspath(os.path.join(HERE, "..", "newsGraph"))
    EDGE_DIR  = os.path.abspath(os.path.join(HERE, "..", "edge_data"))

    GRAPH_IN   = os.path.join(GRAPH_DIR, "graph.pkl")
    COOC_CSV   = os.path.join(EDGE_DIR, "impression_cooccurrence.csv")
    GRAPH_OUT  = os.path.join(GRAPH_DIR, "graph_with_edges.pkl")
    META_OUT   = os.path.join(GRAPH_DIR, "graph_with_edges_meta.json")

    if overwrite:
        GRAPH_OUT = GRAPH_IN
        META_OUT  = os.path.join(GRAPH_DIR, "graph_with_edges_meta.json")

    # ---- load graph ----
    if not os.path.exists(GRAPH_IN):
        raise FileNotFoundError(f"Graph not found: {GRAPH_IN}")
    with open(GRAPH_IN, "rb") as f:
        G = pickle.load(f)
    print(f"[INFO] Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ---- load cooccurrence ----
    if not os.path.exists(COOC_CSV):
        raise FileNotFoundError(f"Cooccurrence CSV not found: {COOC_CSV}")
    df = pd.read_csv(COOC_CSV, dtype={"item_id_a": str, "item_id_b": str, "cooccurrence": int})
    required = {"item_id_a", "item_id_b", "cooccurrence"}
    if not required.issubset(df.columns):
        raise ValueError(f"{COOC_CSV} must have columns: {sorted(required)}; found {list(df.columns)}")

    # Optional filter
    if min_count > 1:
        before = len(df)
        df = df[df["cooccurrence"] >= min_count].copy()
        print(f"[INFO] Filtered by min_count={min_count}: {before} -> {len(df)} rows")

    # ---- add edges ----
    nodes_set = set(G.nodes)
    added, updated, skipped_missing = 0, 0, 0

    print("[INFO] Adding edges with 'cooccurrence' attribute ...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding edges"):
        u, v, c = row["item_id_a"], row["item_id_b"], int(row["cooccurrence"])
        if u not in nodes_set or v not in nodes_set:
            skipped_missing += 1
            continue
        if G.has_edge(u, v):
            G[u][v]["cooccurrence"] = c   # update value
            updated += 1
        else:
            G.add_edge(u, v, cooccurrence=c)  # new edge with attribute
            added += 1

    # ---- save graph ----
    with open(GRAPH_OUT, "wb") as f:
        pickle.dump(G, f)

    total_edges = G.number_of_edges()
    print(f"[DONE] Saved -> {GRAPH_OUT}")
    print(f"       Nodes: {G.number_of_nodes()} | Edges: {total_edges}")
    print(f"       Added edges: {added}, Updated: {updated}, Skipped (missing nodes): {skipped_missing}")

    # ---- metadata ----
    meta = {
        "graph_file": os.path.basename(GRAPH_OUT),
        "nodes": G.number_of_nodes(),
        "edges": total_edges,
        "added_edges": added,
        "updated_edges": updated,
        "skipped_missing": skipped_missing,
        "edge_features": ["cooccurrence"],
        "min_count": min_count,
        "note": "Each edge has attribute 'cooccurrence' = number of times the two items co-occurred."
    }
    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Metadata saved -> {META_OUT}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Add edges to graph from impression_cooccurrence.csv")
    ap.add_argument("--min_count", type=int, default=1,
                    help="Only add pairs with cooccurrence >= this value (default: 1)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite newsGraph/graph.pkl instead of writing graph_with_edges.pkl")
    args = ap.parse_args()
    main(min_count=args.min_count, overwrite=args.overwrite)
