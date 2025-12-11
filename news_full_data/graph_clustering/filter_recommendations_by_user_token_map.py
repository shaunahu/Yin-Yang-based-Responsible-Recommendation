#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter saved_clusters/rs_results/recommendations.csv AND recommendations.pkl
→ Keep only rows / entries whose 'user_id' exists in the VALUES of user_token_map.json
"""

import os
import json
import pandas as pd
import pickle

# --- Paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_clusters", "rs_results"))
CSV_PATH = os.path.join(BASE_DIR, "recommendations.csv")
PKL_PATH = os.path.join(BASE_DIR, "recommendations.pkl")
MAP_PATH = os.path.join(BASE_DIR, "user_token_map.json")
CSV_OUTPUT = os.path.join(BASE_DIR, "recommendations_filtered.csv")
PKL_OUTPUT = os.path.join(BASE_DIR, "recommendations_filtered.pkl")

def load_user_tokens():
    """Load user_token_map.json and return a set of valid numeric user IDs."""
    with open(MAP_PATH, "r", encoding="utf-8") as f:
        user_token_map = json.load(f)
    valid_user_ids = set(map(int, user_token_map.values()))
    print(f"[INFO] Loaded {len(valid_user_ids)} valid numeric user IDs from user_token_map.json")
    return valid_user_ids


def filter_csv(valid_user_ids):
    """Filter recommendations.csv based on numeric user_id."""
    if not os.path.exists(CSV_PATH):
        print(f"[WARN] CSV file not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    if "user_id" not in df.columns:
        raise KeyError(f"'user_id' column not found. Columns: {list(df.columns)}")

    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    before = len(df)
    df_filtered = df[df["user_id"].isin(valid_user_ids)]
    after = len(df_filtered)

    print(f"[INFO] CSV filtered {before} → {after} rows ({after/before:.2%} kept)")
    df_filtered.to_csv(CSV_OUTPUT, index=False)
    print(f"✅ Saved filtered CSV to: {CSV_OUTPUT}")


def filter_pkl(valid_user_ids):
    """Filter recommendations.pkl based on numeric user_id in its structure."""
    if not os.path.exists(PKL_PATH):
        print(f"[WARN] PKL file not found: {PKL_PATH}")
        return

    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    before = 0
    after = 0

    # Case 1: dict {user_id: recommendations}
    if isinstance(data, dict):
        before = len(data)
        filtered_data = {uid: rec for uid, rec in data.items() if int(uid) in valid_user_ids}
        after = len(filtered_data)

    # Case 2: list of dicts [{"user_id": 123, ...}, ...]
    elif isinstance(data, list):
        before = len(data)
        filtered_data = [d for d in data if isinstance(d, dict)
                         and int(d.get("user_id", -1)) in valid_user_ids]
        after = len(filtered_data)
    else:
        raise TypeError(f"Unsupported PKL structure: {type(data)}")

    print(f"[INFO] PKL filtered {before} → {after} entries ({after/before:.2%} kept)")

    with open(PKL_OUTPUT, "wb") as f:
        pickle.dump(filtered_data, f)

    print(f"✅ Saved filtered PKL to: {PKL_OUTPUT}")


def main():
    valid_user_ids = load_user_tokens()
    filter_csv(valid_user_ids)
    filter_pkl(valid_user_ids)


if __name__ == "__main__":
    main()
