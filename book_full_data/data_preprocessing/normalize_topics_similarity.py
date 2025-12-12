#!/usr/bin/env python3
"""
Normalize a movie-topic similarity file.

Assumes an input text file where each non-empty line looks like:

    movie_id_1<TAB>movie_id_2<TAB>similarity_score

For example:
    M1    M2    0.73
    M1    M3    0.50

The script:
1. Reads all similarity scores (last column).
2. Finds min and max scores.
3. Normalizes each score to [0, 1] via:
       (score - min_score) / (max_score - min_score)
4. Writes a new file with the same columns, but the last column is normalized.

Directory layout (example):

movie_full_data/
│
├── data_movie/
│   ├── items.tsv
│   ├── movie-similarity.txt           <-- input
│   ├── new_behaviors.tsv
│
└── data_preprocessing/
    ├── normalize_movies_topics_similarity.py  <-- this file

You can reuse this script by only changing the CONFIG section below.
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple


# ============================================================
# CONFIG: change these for other projects / file names only
# ============================================================

@dataclass
class FileConfig:
    # Name of the folder that contains the data files,
    # relative to the project root (the parent of this script's folder).
    data_dir_name: str = "data_book"

    # Input similarity filename (inside data_dir_name).
    input_similarity_filename: str = "book-similarity.txt"

    # Output filename for the normalized similarity (inside data_dir_name).
    output_similarity_filename: str = "book-similarity-normalized.txt"


CONFIG = FileConfig()


# ============================================================
# PATH HELPERS
# ============================================================

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_data_paths(config: FileConfig) -> Tuple[Path, Path]:
    project_root = get_project_root()
    data_dir = project_root / config.data_dir_name

    input_path = data_dir / config.input_similarity_filename
    output_path = data_dir / config.output_similarity_filename

    return input_path, output_path


# ============================================================
# CORE NORMALIZATION LOGIC
# ============================================================

def read_similarity_file(path: Path) -> List[Tuple[List[str], float]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records: List[Tuple[List[str], float]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                # skip empty lines
                continue
            if stripped.startswith("#"):
                records.append(([stripped], float("nan")))
                continue

            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Line {line_no} in {path} has fewer than 2 columns: {stripped!r}"
                )

            try:
                score = float(parts[-1])
            except ValueError as e:
                raise ValueError(
                    f"Last column is not a float on line {line_no} in {path}: {stripped!r}"
                ) from e

            records.append((parts, score))

    return records


def compute_min_max(scores: List[float]) -> Tuple[float, float]:
    valid_scores = [s for s in scores if s == s]  # s == s filters out NaN
    if not valid_scores:
        raise ValueError("No valid numeric scores found to normalize.")

    return min(valid_scores), max(valid_scores)


def normalize_score(score: float, min_s: float, max_s: float) -> float:
    if score != score:  
        return score

    if max_s == min_s:
        return 0.0

    return (score - min_s) / (max_s - min_s)


def write_normalized_file(
    records: List[Tuple[List[str], float]],
    output_path: Path,
    min_s: float,
    max_s: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for columns, score in records:
            # Comment line: stored as columns=["#comment text"], NaN score
            if len(columns) == 1 and columns[0].startswith("#") and score != score:
                f.write(columns[0] + "\n")
                continue

            norm = normalize_score(score, min_s, max_s)
            # Replace last column with normalized score (as float with 6 decimals)
            columns_out = columns[:-1] + [f"{norm:.6f}"]
            f.write("\t".join(columns_out) + "\n")


def normalize_similarity_file(config: FileConfig) -> None:
    input_path, output_path = get_data_paths(config)

    print(f"[INFO] Project root: {get_project_root()}")
    print(f"[INFO] Reading similarity from: {input_path}")
    print(f"[INFO] Normalized output will be saved to: {output_path}")

    records = read_similarity_file(input_path)
    scores = [score for _, score in records]

    min_s, max_s = compute_min_max(scores)
    print(f"[INFO] Min score: {min_s}")
    print(f"[INFO] Max score: {max_s}")

    write_normalized_file(records, output_path, min_s, max_s)
    print("[INFO] Normalization complete.")


if __name__ == "__main__":
    # You can optionally add argparse here for CLI overrides,
    # but for reusability you only need to edit CONFIG above.
    normalize_similarity_file(CONFIG)
