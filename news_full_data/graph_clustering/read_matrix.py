#!/usr/bin/env python3
# read_matrix.py — load cluster_matrix and manifest, query item_id <-> cluster_id

import os
import json
import argparse
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True, help="Path to cluster_matrix_K*_topk*.pt")
    ap.add_argument("--manifest", required=True, help="Path to cluster_matrix_manifest_K*_topk*.json")
    ap.add_argument("--query_item", help="Query which cluster this item_id belongs to")
    ap.add_argument("--query_cluster", type=int, help="Query all items in this cluster_id")
    args = ap.parse_args()

    # ---- load matrix ----
    M = torch.load(args.matrix)
    print(f"[INFO] Matrix loaded: type={type(M)}, shape={M.shape}")

    # ---- load manifest ----
    with open(args.manifest, "r") as f:
        meta = json.load(f)
    item_to_cluster = meta.get("item_to_cluster", {})
    cluster_to_items = meta.get("cluster_to_items", {})

    # ---- query by item ----
    if args.query_item:
        cid = item_to_cluster.get(args.query_item)
        if cid is None:
            print(f"[QUERY] item_id {args.query_item} not found in manifest.")
        else:
            print(f"[QUERY] item_id {args.query_item} belongs to cluster {cid}")

    # ---- query by cluster ----
    if args.query_cluster is not None:
        items = cluster_to_items.get(str(args.query_cluster), [])
        print(f"[QUERY] cluster {args.query_cluster} has {len(items)} items")
        if len(items) < 20:  # only print small clusters
            print(items[:20])
        else:
            print("... (too many items to print)")

if __name__ == "__main__":
    main()

# # 查 item_id=12345 属于哪个 cluster
# python read_matrix.py \
#   --matrix ../saved_clusters/cluster_matrix_K5_topk5.pt \
#   --manifest ../saved_clusters/cluster_matrix_manifest_K5_topk5.json \
#   --query_item 12345

# # 查 cluster 0 里的所有 item_id
# python read_matrix.py \
#   --matrix ../saved_clusters/cluster_matrix_K5_topk5.pt \
#   --manifest ../saved_clusters/cluster_matrix_manifest_K5_topk5.json \
#   --query_cluster 0

