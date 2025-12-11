#!/usr/bin/env python3
"""
Filter a TSV file to keep only specific columns.

Reusable: You only need to modify INPUT_FILE, OUTPUT_FILE, and KEEP_COLUMNS.
"""

import csv
from pathlib import Path


# ============================================================
# CONFIG — modify only these values
# ============================================================

INPUT_FILE = Path("../data_movie/items.tsv")      # original TSV
OUTPUT_FILE = Path("../data_movie/items_filtered.tsv")  # output TSV

# Keep these 4 columns (order preserved)
KEEP_COLUMNS = ["movieid", "cat1", "abstract", "summary"]


# ============================================================
# MAIN LOGIC
# ============================================================

def filter_columns(input_path: Path, output_path: Path, keep_cols):
    """
    Reads TSV, keeps only selected columns, writes new TSV.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input TSV not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f_in, \
         output_path.open("w", encoding="utf-8", newline="") as f_out:

        reader = csv.DictReader(f_in, delimiter="\t")
        missing = [c for c in keep_cols if c not in reader.fieldnames]

        if missing:
            raise ValueError(f"ERROR: Missing required columns: {missing}")

        writer = csv.DictWriter(f_out, fieldnames=keep_cols, delimiter="\t")
        writer.writeheader()

        for row in reader:
            filtered = {col: row[col] for col in keep_cols}
            writer.writerow(filtered)

    print(f"[INFO] Wrote filtered TSV: {output_path}")


if __name__ == "__main__":
    filter_columns(INPUT_FILE, OUTPUT_FILE, KEEP_COLUMNS)
