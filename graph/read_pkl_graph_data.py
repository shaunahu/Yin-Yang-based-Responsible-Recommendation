import pickle
import networkx as nx

def read_graph_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        graph = pickle.load(f)
    return graph

if __name__ == "__main__":
    # Adjust the path as necessary based on your working directory
    pkl_path = "graph_data/news_item_graph_full_augmented.pkl" # with edge info
    # pkl_path = "graph_data/news_item_subset_300_graph.pkl"

    graph = read_graph_from_pkl(pkl_path)
    
    print("Graph loaded successfully!")
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    # Print a few nodes and edges as a sample
    print("Sample nodes:", list(graph.nodes(data=True))[:20])
    print("Sample edges:", list(graph.edges(data=True))[:20])
