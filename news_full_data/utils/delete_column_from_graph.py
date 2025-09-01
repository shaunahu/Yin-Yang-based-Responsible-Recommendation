# delete_column_from_graph.py
import os
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "news_full_dataset_graph_data"))

FEATURE_NAME = "impression_cooccurrence_prob"  # feature to delete

def main():
    meta_path  = os.path.join(GRAPH_DIR, "features_meta.json")
    feats_path = os.path.join(GRAPH_DIR, "edge_feats.npy")
    nodes_path = os.path.join(GRAPH_DIR, "nodes.npy")

    print(f"[INFO] Using GRAPH_DIR: {GRAPH_DIR}")
    print(f"[INFO] Feature to delete: {FEATURE_NAME}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    if FEATURE_NAME not in meta.get("features", {}):
        print(f"[INFO] Feature '{FEATURE_NAME}' not in meta. Nothing to delete.")
        return

    col = meta["features"].pop(FEATURE_NAME)
    print(f"[INFO] Removing feature '{FEATURE_NAME}' from column {col}")

    # Load memmap and overwrite column with NaN
    nodes = np.load(nodes_path, allow_pickle=True)
    n = len(nodes); m = n * (n - 1) // 2
    num_cols = meta["num_cols"]

    feat_mm = np.memmap(feats_path, dtype=np.float16, mode="r+", shape=(m, num_cols))
    feat_mm[:, col] = np.nan
    feat_mm.flush()

    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Feature '{FEATURE_NAME}' deleted. Column {col} is now free.")

if __name__ == "__main__":
    main()
