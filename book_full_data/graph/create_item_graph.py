# create_item_graph.py
# Sep 19, 2025
import os
import pandas as pd
import networkx as nx
from tqdm import tqdm
import pickle


def main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    # TSV for movies
    DATA_TSV = os.path.abspath(os.path.join(HERE, "..", "data_book", "items_filtered.tsv"))
    OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "booksGraph"))
    PKL_PATH = os.path.join(OUT_DIR, "graph.pkl")

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[INFO] Loading items from: {DATA_TSV}")
    # IMPORTANT: sep="\t" for TSV
    df = pd.read_csv(DATA_TSV, dtype=str, sep="\t")

    # --- Figure out which ID column to use ---
    if "item_id" in df.columns:
        id_col = "item_id"
    elif "movieid" in df.columns:
        id_col = "movieid"
    else:
        raise ValueError(
            f"Expected an ID column 'item_id' or 'movieid' in {DATA_TSV}, "
            f"found: {list(df.columns)}"
        )

    # --- Figure out which columns to treat as topic/title/abstract ---
    # For news-style: topic / title / abstract
    # For movie-style: cat1 (topic-ish) / summary (title-ish) / abstract
    topic_col = None
    title_col = None
    abs_col = None

    if "topic" in df.columns:
        topic_col = "topic"
    elif "cat1" in df.columns:
        topic_col = "cat1"

    if "title" in df.columns:
        title_col = "title"
    elif "summary" in df.columns:
        title_col = "summary"

    if "abstract" in df.columns:
        abs_col = "abstract"

    print("[INFO] Column mapping:")
    print(f"       id_col    = {id_col}")
    print(f"       topic_col = {topic_col}")
    print(f"       title_col = {title_col}")
    print(f"       abs_col   = {abs_col}")

    G = nx.Graph()

    print("[INFO] Adding nodes (no edges yet).")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding nodes"):
        item_id = row[id_col]
        attrs = {}

        if topic_col is not None:
            val = row[topic_col]
            attrs["topic"] = val if (pd.notna(val)) else ""

        if title_col is not None:
            val = row[title_col]
            attrs["title"] = val if (pd.notna(val)) else ""

        if abs_col is not None:
            val = row[abs_col]
            attrs["abstract"] = val if (pd.notna(val)) else ""

        G.add_node(item_id, **attrs)

    print(f"[INFO] Graph summary: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    with open(PKL_PATH, "wb") as f:
        pickle.dump(G, f)
    print(f"[INFO] Saved NetworkX graph -> {PKL_PATH}")


if __name__ == "__main__":
    main()
