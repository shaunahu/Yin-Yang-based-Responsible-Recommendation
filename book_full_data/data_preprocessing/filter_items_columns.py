#!/usr/bin/env python3
# filter_books_items.py
# Keep only columns 1,2,4,5 and rename them for the BOOK dataset.

import os
import pandas as pd

def main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    INPUT = os.path.join(HERE, "../data_book/items.tsv")               # adjust if needed
    OUTPUT = os.path.join(HERE, "../data_book/items_filtered.tsv")

    # Load TSV with NO header
    df = pd.read_csv(INPUT, sep="\t", header=None, dtype=str)

    # Select columns 1,2,4,5 → 0-indexed: col 0,1,3,4
    df_small = df[[0, 1, 3, 4]].copy()

    # Rename columns
    df_small.columns = ["bookid", "cat", "abstract", "summary"]

    # Save filtered TSV
    df_small.to_csv(OUTPUT, sep="\t", index=False)

    print(f"[DONE] Saved filtered file → {OUTPUT}")
    print(f"[INFO] Rows: {len(df_small)} | Columns: {df_small.columns.tolist()}")

if __name__ == "__main__":
    main()
