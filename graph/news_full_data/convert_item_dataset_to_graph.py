# convert_item_dataset_to_graph.py
# too slow deprecated
# Use convert_item_dataset_to_graph_memmap.py instead
import os
import pandas as pd
import networkx as nx
import pickle
from itertools import combinations
from tqdm import tqdm  # ✅ 新增

def build_fully_connected_graph_from_csv(csv_path: str, id_col: str = "item_id") -> nx.Graph:
    """
    Load item data from a CSV file and build a fully connected undirected graph.
    Assumes the CSV contains a column with item IDs (default: 'item_id').
    """
    df = pd.read_csv(csv_path, dtype=str)
    if id_col not in df.columns:
        raise ValueError(f"Expected column '{id_col}' in {csv_path}. Found columns: {list(df.columns)}")

    items = df[id_col].dropna().unique().tolist()

    G = nx.Graph()
    G.add_nodes_from(items)

    # Generate edge with progress bar
    total_edges = len(items) * (len(items) - 1) // 2
    print(f"👉 Building fully connected graph: {len(items)} nodes, ~{total_edges:,} edges")
    for u, v in tqdm(combinations(items, 2), total=total_edges, desc="Adding edges"):
        G.add_edge(u, v)

    return G

def save_graph_as_pickle(graph: nx.Graph, output_path: str) -> None:
    with open(output_path, "wb") as f:
        pickle.dump(graph, f)

# def save_graph_as_edgelist(graph: nx.Graph, output_path: str) -> None:
#     nx.write_edgelist(graph, output_path, data=False)

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Input CSV (filtered items)
    input_csv = os.path.join(BASE_DIR, "data_news", "items_filtered.csv")

    # Output folder（existing）
    out_dir = os.path.join(BASE_DIR, "news_full_dataset_graph_data")
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"Output directory not found: {out_dir}")

    # Output files
    pickle_path = os.path.join(out_dir, "items_filtered_graph.pkl")
    # edgelist_path = os.path.join(out_dir, "items_filtered_graph.edgelist")

    # Build & save graph
    graph = build_fully_connected_graph_from_csv(input_csv, id_col="item_id")
    save_graph_as_pickle(graph, pickle_path)
    # save_graph_as_edgelist(graph, edgelist_path)

    print("✅ Graph saved.")
    print(f"   Pickle   : {pickle_path}")
    # print(f"   Edgelist : {edgelist_path}")
    print(f"   Nodes    : {graph.number_of_nodes()}")
    print(f"   Edges    : {graph.number_of_edges()}")

if __name__ == "__main__":
    main()
