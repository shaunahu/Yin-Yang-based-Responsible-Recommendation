import pickle
import networkx as nx

# Path to your saved graph
file_path = "../graph_data/news_item_graph_with_clusters.pkl"

# Load the graph
with open(file_path, "rb") as f:
    G = pickle.load(f)

# Example: inspect the graph
print(f"✅ Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Example: print first 5 nodes with cluster labels
for i, (node, data) in enumerate(G.nodes(data=True)):
    print(f"Node: {node}, Cluster: {data.get('cluster')}")
    if i >= 4:
        break
