import os
import json
import numpy as np
import random

# Resolve BASE_GRAPH_DIR relative to this script's location
BASE_GRAPH_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "news_full_dataset_graph_data")
)

print(f"[INFO] Using BASE_GRAPH_DIR = {BASE_GRAPH_DIR}")


def edge_index_to_pair(n, k):
    """Convert a linear edge index k to a node pair (i, j) in upper-triangular order."""
    cursor = 0
    for i in range(n - 1):
        cnt = n - i - 1
        if k < cursor + cnt:
            j = i + 1 + (k - cursor)
            return i, j
        cursor += cnt
    raise IndexError("Edge index out of range")

def check_random_edges(base_dir=BASE_GRAPH_DIR, num_samples=5):
    # Load node order
    nodes = np.load(os.path.join(base_dir, "nodes.npy"), allow_pickle=True)
    n = len(nodes)

    # Load feature metadata
    with open(os.path.join(base_dir, "features_meta.json")) as f:
        meta = json.load(f)
    features = meta["features"]

    # Load edge feature memmap
    m = n * (n - 1) // 2
    feats = np.memmap(os.path.join(base_dir, "edge_feats.npy"),
                      dtype=np.float16, mode="r",
                      shape=(m, meta["num_cols"]))

    print(f"✅ Total {n} nodes, {m} edges")
    print(f"📊 Feature columns: {features}\n")

    # Randomly sample edges
    sample_indices = random.sample(range(m), num_samples)
    for k in sample_indices:
        i, j = edge_index_to_pair(n, k)
        node_i, node_j = nodes[i], nodes[j]
        values = {name: float(feats[k, col]) for name, col in features.items()}
        print(f"edge_index={k} ({node_i}, {node_j}) -> {values}")

if __name__ == "__main__":
    check_random_edges(num_samples=100)  # Change 5 to any number of samples
