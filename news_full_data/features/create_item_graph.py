# create_item_graph.py
# Sep 19,2025
import os
import pandas as pd
import networkx as nx
from tqdm import tqdm
import pickle

def main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    DATA_CSV = os.path.abspath(os.path.join(HERE, "..", "data_news", "items_filtered.csv"))
    OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "newsGraph"))
    PKL_PATH = os.path.join(OUT_DIR, "graph.pkl")

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[INFO] Loading items from: {DATA_CSV}")
    df = pd.read_csv(DATA_CSV, dtype=str)

    if "item_id" not in df.columns:
        raise ValueError(f"Expected column 'item_id' in {DATA_CSV}, found: {list(df.columns)}")

    G = nx.Graph()

    has_topic = "topic" in df.columns
    has_title = "title" in df.columns
    has_abs   = "abstract" in df.columns

    print("[INFO] Adding nodes (no edges yet).")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding nodes"):
        item_id = row["item_id"]
        attrs = {}
        if has_topic: attrs["topic"] = row["topic"] if pd.notna(row["topic"]) else ""
        if has_title: attrs["title"] = row["title"] if pd.notna(row["title"]) else ""
        if has_abs:   attrs["abstract"] = row["abstract"] if pd.notna(row["abstract"]) else ""
        G.add_node(item_id, **attrs)

    print(f"[INFO] Graph summary: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    with open(PKL_PATH, "wb") as f:
        pickle.dump(G, f)
    print(f"[INFO] Saved NetworkX graph -> {PKL_PATH}")

if __name__ == "__main__":
    main()
