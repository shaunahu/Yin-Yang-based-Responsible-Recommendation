import pickle
import networkx as nx
from graph_clustering_adaptive_k import GraphClusterer

def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_graph(graph, path):
    with open(path, "wb") as f:
        pickle.dump(graph, f)

def main():
    # === Step 1: Load the graph ===
    input_path = "../graph_data/news_item_graph_full_augmented_with_frequency.pkl"
    G = load_graph(input_path)
    print(f"✅ Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # === Step 2: Define feature weights ===
    weights = {
        'semantic_similarity': 0.4,
        'topic_similarity': 0.3,
        'sentiment_similarity': 0.2,
        'frequent': 0.1
    }

    # === Step 3: Run clustering ===
    clusterer = GraphClusterer(G)
    result = clusterer.cluster(num_clusters=5, feature_weights=weights, verbose=True)

    # === Step 4: Annotate nodes with cluster labels ===
    label_map = result['label_map']
    for node, label in label_map.items():
        G.nodes[node]['cluster'] = label
    print("🟢 Cluster labels assigned to nodes")

    # === Step 5: Save updated graph ===
    output_path = "../graph_data/news_item_graph_with_clusters.pkl"
    save_graph(G, output_path)
    print(f"📦 Graph with clusters saved to: {output_path}")

if __name__ == "__main__":
    main()
