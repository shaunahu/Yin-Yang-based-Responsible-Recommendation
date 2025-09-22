#!/usr/bin/env python3
# cluster_reader.py — reusable class to load cluster_matrix and manifest

import os
import json
import torch

class ClusterReader:
    def __init__(self, matrix_path: str, manifest_path: str):
        # ---- load matrix ----
        if not os.path.exists(matrix_path):
            raise FileNotFoundError(f"Matrix file not found: {matrix_path}")
        self.matrix = torch.load(matrix_path)
        print(f"[INFO] Matrix loaded: type={type(self.matrix)}, shape={self.matrix.shape}")

        # ---- load manifest ----
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        with open(manifest_path, "r") as f:
            meta = json.load(f)

        self.item_to_cluster = meta.get("item_to_cluster", {})
        self.cluster_to_items = meta.get("cluster_to_items", {})
        self.num_clusters = len(self.cluster_to_items)

    def get_cluster_of_item(self, item_id: str):
        """Return cluster_id of a given item_id (or None if not found)."""
        return self.item_to_cluster.get(item_id)

    def get_items_in_cluster(self, cluster_id: int):
        """Return list of all item_ids in a given cluster."""
        return self.cluster_to_items.get(str(cluster_id), [])

    def get_cluster_vector(self, cluster_id: int):
        """Return the cluster row vector from the PyTorch matrix."""
        if cluster_id < 0 or cluster_id >= self.num_clusters:
            raise ValueError(f"Cluster id {cluster_id} out of range (0..{self.num_clusters-1})")
        return self.matrix[cluster_id]

    def summary(self):
        """Print a short summary of clusters."""
        print(f"[SUMMARY] {self.num_clusters} clusters available.")
        for cid, items in self.cluster_to_items.items():
            print(f" - Cluster {cid}: {len(items)} items")

# ---------------- Example Usage ----------------
if __name__ == "__main__":
    # 手动改成你的路径
    matrix_path = "../saved_clusters/cluster_matrix_K5_topk5.pt"
    manifest_path = "../saved_clusters/cluster_matrix_manifest_K5_topk5.json"

    reader = ClusterReader(matrix_path, manifest_path)
    reader.summary()

    # 查 item_id 属于哪个 cluster
    item_id = "12345"
    print(f"Item {item_id} -> cluster {reader.get_cluster_of_item(item_id)}")

    # 查 cluster 0 里的所有 items
    cluster_id = 0
    items = reader.get_items_in_cluster(cluster_id)
    print(f"Cluster {cluster_id} has {len(items)} items, first few: {items[:10]}")

    # 取 cluster 0 的向量
    vec = reader.get_cluster_vector(cluster_id)
    print(f"Cluster {cluster_id} vector shape: {vec.shape}")
