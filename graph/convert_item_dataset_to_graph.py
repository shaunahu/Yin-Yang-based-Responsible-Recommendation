import pandas as pd
import networkx as nx
import pickle

def build_fully_connected_graph_from_csv(csv_path):
    """
    Load item data from a CSV file and build a fully connected undirected graph.
    """
    df = pd.read_csv(csv_path)
    item_column = df.columns[0]  # assume the first column contains item IDs
    items = df[item_column].dropna().unique()

    G = nx.Graph()
    G.add_nodes_from(items)
    edges = [(i, j) for idx, i in enumerate(items) for j in items[idx + 1:]]
    G.add_edges_from(edges)

    return G

def save_graph_as_pickle(graph, output_path):
    """
    Save a NetworkX graph as a pickle file.
    """
    with open(output_path, 'wb') as f:
        pickle.dump(graph, f)

def save_graph_as_edgelist(graph, output_path):
    """
    Save a NetworkX graph in edgelist text format.
    """
    nx.write_edgelist(graph, output_path, data=False)

def main():
    # Define input and output paths
    csv_path = "./data/news_item_subset_300.csv"
    pickle_path = "./graph_data/news_item_subset_300_graph.pkl"
    edgelist_path = "./graph_data/news_item_subset_300_graph.edgelist"

    # Build the graph
    graph = build_fully_connected_graph_from_csv(csv_path)

    # Save graph in both formats
    save_graph_as_pickle(graph, pickle_path)
    save_graph_as_edgelist(graph, edgelist_path)

    print("Graph has been saved as .pkl and .edgelist formats.")
    print("Nodes:", graph.number_of_nodes(), ", Edges:", graph.number_of_edges())

if __name__ == "__main__":
    main()
