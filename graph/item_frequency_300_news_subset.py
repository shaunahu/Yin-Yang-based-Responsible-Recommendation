import pandas as pd
import networkx as nx
import pickle
from collections import defaultdict
from tqdm import tqdm  # 进度条模块

# === File paths ===
impression_path = "./data/filtered_news_behaviors_300_subset.csv"
graph_subset_path = "./graph_data/news_item_subset_300_graph.pkl"
graph_augmented_path = "./graph_data/news_item_graph_full_augmented.pkl"
output_graph_path = "./graph_data/news_item_graph_full_augmented_with_frequency.pkl"

# === Step 1: Load impression data ===
impression_df = pd.read_csv(impression_path)

# === Step 2: Load subset graph for edge list to monitor ===
with open(graph_subset_path, "rb") as f:
    G_subset = pickle.load(f)
edges = list(G_subset.edges())

# === Step 3: Count edge frequencies with tqdm progress bar ===
edge_frequency = defaultdict(int)

print("Processing impression sequences to count edge frequencies...\n")
for impression in tqdm(impression_df['impression'], desc="Processing impressions"):
    items = [x.split('-')[0] for x in str(impression).split()]
    item_index = {item: idx for idx, item in enumerate(items)}
    for u, v in edges:
        if u in item_index and v in item_index and item_index[u] < item_index[v]:
            edge_frequency[(u, v)] += 1

# === Step 4: Load full augmented graph and inject frequency info ===
with open(graph_augmented_path, "rb") as f:
    G_augmented = pickle.load(f)

for (u, v), freq in edge_frequency.items():
    if G_augmented.has_edge(u, v):
        G_augmented[u][v]['frequent'] = freq

# === Step 5: Save updated graph ===
with open(output_graph_path, "wb") as f:
    pickle.dump(G_augmented, f)

print(f"\n✅ Updated graph saved to: {output_graph_path}")

# === Step 6: Show top 10 edges with frequency attribute for verification ===
print("\nTop 10 edges with non-zero 'frequent' attribute:\n")
count = 0
for u, v, attrs in G_augmented.edges(data=True):
    if 'frequent' in attrs and attrs['frequent'] > 0:
        print(f"{u} -> {v}, attributes: {attrs}")
        count += 1
    if count >= 10:
        break
