import pickle
import networkx as nx

def load_graph(graph_path):
    """Load a graph object from a pickle file."""
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    return G

def preview_graph(G, num_edges=5):
    """Print a preview of a few edges and their attributes."""
    edges = list(G.edges(data=True))
    for edge in edges[:num_edges]:
        print(edge)

if __name__ == "__main__":
    graph_path = './graph_data/news_item_graph_full_augmented.pkl'
    G = load_graph(graph_path)
    preview_graph(G)
