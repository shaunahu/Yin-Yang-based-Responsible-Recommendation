# Re-import after kernel reset
# Convert to pickle data format
import pandas as pd
import networkx as nx
import pickle

# Reload the CSV
df = pd.read_csv('./data/subset_300.csv')

# Clean and extract item IDs
def extract_items(entry):
    if pd.isna(entry):
        return []
    return [x.split('-')[0] for x in entry.strip().split()]

# Collect all item IDs from both columns
all_items = set()
for col in ['impression', 'history']:
    df[col + '_items'] = df[col].apply(extract_items)
    for items in df[col + '_items']:
        all_items.update(items)

# Build fully connected undirected graph
G = nx.Graph()
G.add_nodes_from(all_items)
edges = [(i, j) for idx, i in enumerate(all_items) for j in list(all_items)[idx+1:]]
G.add_edges_from(edges)

# Print graph info
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# Save the graph as .pkl
pkl_path = "./data/item_graph.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(G, f)

print(pkl_path)
