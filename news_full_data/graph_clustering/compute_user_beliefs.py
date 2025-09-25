#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute per-user belief distributions over clusters, average them, and
insert the averaged belief (normalized to sum=1) at the TOP of the
cluster manifest JSON (by writing the new key first).

Inputs:
  --manifest  ../saved_clusters/cluster_matrix_manifest_K5_topk5.json
  --behaviors ../saved_clusters/new_behaviors_filtered.csv
Outputs:
  --output    ../saved_clusters/user_beliefs_K5_topk5.tsv
  (and updates the manifest JSON in place by adding "avg_belief" first)

Belief logic:
- Clicked (-1) items add POSITIVE_WEIGHT to their cluster
- Non-clicked (-0) items add NEGATIVE_WEIGHT (default 0.0, set < 0 to penalize)
- Add ALPHA smoothing to every cluster score
- Normalize each user's vector to sum to 1
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict

# --------------------------
# Hyperparameters
# --------------------------
POSITIVE_WEIGHT = 1.0
NEGATIVE_WEIGHT = -0.0   # set negative (e.g., -0.2) to penalize non-click exposures
ALPHA = 0.01

# --------------------------
# Optional logger fallback
# --------------------------
try:
    from common import logger  # your project logger
except Exception:
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s | %(message)s")
    class _FallbackLogger:
        def info(self, msg): _logging.info(msg)
        def debug(self, msg): _logging.debug(msg)
        def warning(self, msg): _logging.warning(msg)
        def error(self, msg): _logging.error(msg)
    logger = _FallbackLogger()


# --------------------------
# Helpers
# --------------------------
def load_clusters_from_manifest(manifest_path: str):
    """
    Reads cluster_to_items and builds:
      - clusters: {cluster_id(int): set(item_ids)}
      - messages: set(all item_ids)
      - message_cluster_map: {item_id: cluster_id(int)}
      - cluster_ids: sorted list of cluster IDs (ints)
    """
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    if "cluster_to_items" not in manifest:
        raise KeyError("Manifest missing 'cluster_to_items' key")

    cluster_to_items = manifest["cluster_to_items"]
    clusters = {int(cid): set(items) for cid, items in cluster_to_items.items()}

    messages = set()
    message_cluster_map = {}
    for cid, items in clusters.items():
        for it in items:
            messages.add(it)
            message_cluster_map[it] = cid

    cluster_ids = sorted(clusters.keys())
    logger.info(f"Loaded {len(cluster_ids)} clusters, {len(messages)} items from manifest.")
    return manifest, clusters, messages, message_cluster_map, cluster_ids


def split_impressions(df: pd.DataFrame, valid_ids: set) -> pd.DataFrame:
    """
    From 'impression' column (space-separated tokens like 'N123-1', 'N456-0'):
      - Keep only items whose base ID is in valid_ids
      - Create 'impression_1' (clicked) and 'impression_0' (not clicked)
      - Drop rows where both lists are empty
    """
    def process(imp_str):
        if pd.isna(imp_str):
            return pd.Series([[], []])
        tokens = str(imp_str).split()
        filtered = [t for t in tokens if t.split('-')[0] in valid_ids]
        group_0 = [t.split("-0")[0] for t in filtered if t.endswith("-0")]
        group_1 = [t.split("-1")[0] for t in filtered if t.endswith("-1")]
        return pd.Series([group_0, group_1])

    df[["impression_0", "impression_1"]] = df["impression"].apply(process)
    before = len(df)
    df = df[~((df["impression_0"].str.len() == 0) & (df["impression_1"].str.len() == 0))].reset_index(drop=True)
    after = len(df)
    if after < before:
        logger.info(f"Dropped {before - after} rows with no valid impressions.")
    return df


# --------------------------
# Core
# --------------------------
def compute_user_beliefs_and_update_manifest(manifest_path: str,
                                             behaviors_csv: str,
                                             output_tsv: str):
    # 1) Load clusters/items from manifest
    manifest, clusters, messages, message_cluster_map, cluster_ids = load_clusters_from_manifest(manifest_path)
    id_to_idx = {cid: i for i, cid in enumerate(cluster_ids)}
    num_clusters = len(cluster_ids)

    # 2) Read behaviors & split impressions
    users = pd.read_csv(behaviors_csv)
    if "impression" not in users.columns:
        raise KeyError("Input CSV must contain an 'impression' column.")
    users = split_impressions(users, messages)

    # 3) Per-user belief computation
    user_beliefs = np.zeros((len(users), num_clusters), dtype=np.float64)
    belief_list = []

    for idx, row in users[["impression_1", "impression_0"]].iterrows():
        cluster_scores = np.zeros(num_clusters, dtype=np.float64)

        # positive evidence from clicks
        for item_id in row["impression_1"]:
            cid = message_cluster_map.get(item_id)
            if cid is not None:
                cluster_scores[id_to_idx[cid]] += POSITIVE_WEIGHT

        # negative evidence from non-click exposures
        for item_id in row["impression_0"]:
            cid = message_cluster_map.get(item_id)
            if cid is not None:
                cluster_scores[id_to_idx[cid]] += NEGATIVE_WEIGHT

        # smoothing + normalization
        smoothed = cluster_scores + ALPHA
        np.clip(smoothed, 1e-6, None, out=smoothed)
        belief = smoothed / smoothed.sum()

        user_beliefs[idx] = belief
        belief_list.append(belief.tolist())

    users["belief"] = belief_list

    # 4) Average belief across users and normalize (sums to 1)
    avg_belief = user_beliefs.mean(axis=0)
    avg_belief_norm = (avg_belief / max(avg_belief.sum(), 1e-12)).tolist()

    logger.info(f"Average belief across clusters (normalized): {avg_belief_norm}")

    # 5) Save per-user TSV
    os.makedirs(os.path.dirname(output_tsv), exist_ok=True)
    users.to_csv(output_tsv, sep="\t", index=False)
    logger.info(f"User beliefs saved to {output_tsv}")

    # 6) Update manifest JSON: put avg_belief FIRST (top of file)
    new_manifest = OrderedDict()
    new_manifest["avg_belief"] = avg_belief_norm
    new_manifest["avg_belief_cluster_order"] = cluster_ids  # clarify index order
    for k, v in manifest.items():
        new_manifest[k] = v

    with open(manifest_path, "w") as f:
        json.dump(new_manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"Updated manifest with avg_belief at the top: {manifest_path}")

    # Also print the requested lines
    print("Average belief across clusters (normalised):")
    print(avg_belief_norm)


# --------------------------
# CLI
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Compute per-user beliefs and add averaged belief to the manifest JSON (as the first key).")
    parser.add_argument("--manifest",
                        default=os.path.join("..", "saved_clusters", "cluster_matrix_manifest_K5_topk5.json"),
                        help="Path to cluster manifest JSON.")
    parser.add_argument("--behaviors",
                        default=os.path.join("..", "saved_clusters", "new_behaviors_filtered.csv"),
                        help="Path to behaviors CSV containing an 'impression' column.")
    parser.add_argument("--output",
                        default=os.path.join("..", "saved_clusters", "user_beliefs_K5_topk5.tsv"),
                        help="Path to output TSV with per-user beliefs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compute_user_beliefs_and_update_manifest(args.manifest, args.behaviors, args.output)



#  python3 compute_user_beliefs.py \
#   --manifest ../saved_clusters/cluster_matrix_manifest_K5_topk5.json \
#   --behaviors ../data_news/new_behaviors_filtered.csv \
#   --output ../saved_clusters/user_beliefs_K5_topk5.tsv