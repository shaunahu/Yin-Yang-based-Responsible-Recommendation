#!/usr/bin/env python3
# impression_cooccurrence_movies.py
# Count co-occurrence of movies in impression lists (MOVIES dataset).
# Dec 11, 2025

import os
import json
import numpy as np
import pandas as pd
from itertools import combinations
from tqdm import tqdm
from collections import Counter

# --- paths (relative to this script) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# For movies: impressions are in data_movie/new_behaviors.tsv
BEHAVIORS_FILE = os.path.abspath(
    os.path.join(BASE_DIR, "..", "data_movie", "new_behaviors.tsv")
)

OUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "edge_data"))
OUT_CSV = os.path.join(OUT_DIR, "impression_cooccurrence_movies.csv")

CHUNK_SIZE = 100_000  # adjust for memory/speed


def parse_impression(cell: str):
    """
    Parse one impression cell into a de-duplicated, sorted list of movie IDs.

    Expected format per cell (MOVIES dataset):
        "mid1-1 mid2-0 mid3-1 ..."
    or with commas:
        "mid1-1,mid2-0,mid3-1"

    We:
      - split on comma/whitespace
      - keep the part before the first '-' as movieid
      - de-duplicate per row
      - sort so (a,b) is stable with a < b
    """
    if not isinstance(cell, str) or not cell.strip():
        return []
    toks = cell.replace(",", " ").split()
    ids = [t.split("-", 1)[0] for t in toks if t]
    return sorted(set(ids))


def count_pair_frequencies(file_path: str, chunk_size: int = CHUNK_SIZE):
    """
    Return Counter((a,b)->count) where a<b are movie IDs that appeared together
    in a single 'impression' row; also return total number of non-empty rows.
    """
    pair_counts = Counter()
    total_rows = 0

    print(f"[INFO] Scanning impressions from: {file_path}")

    # Choose separator based on extension: TSV vs CSV
    if file_path.endswith(".tsv"):
        reader = pd.read_csv(file_path, dtype=str, chunksize=chunk_size, sep="\t")
    else:
        reader = pd.read_csv(file_path, dtype=str, chunksize=chunk_size)

    for chunk in tqdm(reader, desc="Counting co-occurrences"):
        # normalize column name if needed
        if "impression" not in chunk.columns:
            lower = {c.lower(): c for c in chunk.columns}
            if "impression" in lower:
                chunk = chunk.rename(columns={lower["impression"]: "impression"})
            else:
                raise ValueError("Column 'impression' not found in behaviors file.")

        for cell in chunk["impression"].fillna(""):
            items = parse_impression(cell)
            if not items:
                continue
            total_rows += 1
            if len(items) < 2:
                continue
            # count unique pairs within the row
            for a, b in combinations(items, 2):
                pair_counts[(a, b)] += 1

    print(
        f"[INFO] Counted {total_rows} non-empty impressions; "
        f"{len(pair_counts):,} unique co-occurring movie pairs."
    )
    return pair_counts, total_rows


def save_counts_to_csv(pair_counts: Counter, out_csv: str):
    """Save pairs with count >= 1 to CSV."""
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Build DataFrame
    rows = ((a, b, c) for (a, b), c in pair_counts.items() if c >= 1)
    df = pd.DataFrame(rows, columns=["movieid_a", "movieid_b", "cooccurrence"])

    # Sort (highest cooccurrence first), then lexicographically
    if not df.empty:
        df = df.sort_values(
            by=["cooccurrence", "movieid_a", "movieid_b"],
            ascending=[False, True, True],
        )

    df.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote {len(df):,} rows -> {out_csv}")


def main():
    pair_counts, _ = count_pair_frequencies(BEHAVIORS_FILE)
    save_counts_to_csv(pair_counts, OUT_CSV)


if __name__ == "__main__":
    main()
