import pickle
from graph_clustering_adaptive_k import GraphClusterer

# Load the graph
with open("../graph_data/news_item_graph_full_augmented_with_frequency.pkl", "rb") as f:
    G = pickle.load(f)

# Define your custom feature weights
weights = {
    'semantic_similarity': 0.4,
    'topic_similarity': 0.3,
    'sentiment_similarity': 0.2,
    'frequent': 0.1  # optional; can be removed or set to 0
}

# Run clustering
clusterer = GraphClusterer(G)
result = clusterer.cluster(num_clusters=5, feature_weights=weights, verbose=True)

# (Optional) Access and use result
# For example:

# print(result['label_map']) #gives you {node_id: cluster_label}
# print(result['clusters']) # gives you details per cluster
