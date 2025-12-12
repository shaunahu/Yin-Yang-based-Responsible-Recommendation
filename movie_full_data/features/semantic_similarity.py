#!/usr/bin/env python3
# semantic_similarity_books.py
# Add semantic similarity (summary + abstract) for BOOKS dataset edges
# Dec 12, 2025

import os
import json
import pickle
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer


# -------------------------------------------------------------------
# Load graph
# -------------------------------------------------------------------
def load_graph(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# -------------------------------------------------------------------
# Build node texts from summary + abstract
# -------------------------------------------------------------------
def build_node_texts(G, items_tsv: str) -> Tuple[List[str], List[str]]:
    """
    For BOOKS dataset:
        text = summary + " " + abstract

    items_filtered.tsv must contain:
        bookid, summary, abstract
    """
    nodes = list(G.nodes())

    if not os.path.exists(items_tsv):
        raise FileNotFoundError(f"Items TSV not found: {items_tsv}")

    df = pd.read_csv(items_tsv, sep="\t", dtype=str)

    required = {"bookid", "summary", "abstract"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{items_tsv} missing columns: {missing}")

    # Build lookup dict
    text_lookup = {}
    for _, row in df.iterrows():
        bid = str(row["bookid"])
        summary = str(row["summary"]) if pd.notna(row["summary"]) else ""
        abstract = str(row["abstract"]) if pd.notna(row["abstract"]) else ""
        text_lookup[bid] = (summary + " " + abstract).strip().lower()

    texts = []
    missing = 0
    for n in nodes:
        txt = text_lookup.get(str(n), "")
        if txt == "":
            missing += 1
        texts.append(txt)

    if missing > 0:
        print(f"[WARN] {missing} book nodes have no text in items_filtered.tsv")

    return nodes, texts


# -------------------------------------------------------------------
# Compute sentence-transformer embeddings
# -------------------------------------------------------------------
def compute_embeddings(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading model: {model_name} ({device})")

    model = SentenceTransformer(model_name, device=device)

    print(f"[INFO] Encoding {len(texts)} texts → embeddings...")
    with torch.inference_mode():
        emb = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            device=device,
            normalize_embeddings=True,
        )

    return emb.cpu().numpy().astype(np.float32, copy=False)


# -------------------------------------------------------------------
# Write semantic similarity to edges
# -------------------------------------------------------------------
def write_semantic_to_edges(G, nodes, emb, feature_name: str) -> int:
    node2idx = {nid: i for i, nid in enumerate(nodes)}
    updated = 0

    edges = list(G.edges())
    print(f"[INFO] Updating {len(edges):,} edges with '{feature_name}'...")

    for u, v in tqdm(edges, desc="semantic_similarity"):
        iu = node2idx.get(u)
        iv = node2idx.get(v)

        if iu is None or iv is None:
            continue

        cos = float(np.dot(emb[iu], emb[iv]))  # cosine similarity [-1,1]
        sim01 = (cos + 1.0) * 0.5              # normalize to [0,1]

        G[u][v][feature_name] = sim01
        updated += 1

    return updated


# -------------------------------------------------------------------
# Update metadata JSON
# -------------------------------------------------------------------
def update_meta(meta_path, graph_filename, G, feature_name, updated_count, model_name, batch_size):
    import datetime

    if os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path))
        except:
            meta = {}
    else:
        meta = {}

    meta.setdefault("graph_file", graph_filename)
    meta["nodes"] = G.number_of_nodes()
    meta["edges"] = G.number_of_edges()

    feats = meta.get("edge_features", [])
    if feature_name not in feats:
        feats.append(feature_name)
    meta["edge_features"] = feats

    efc = meta.get("edge_feature_counts", {})
    efc[feature_name] = updated_count
    meta["edge_feature_counts"] = efc

    meta.setdefault("models", {})
    meta["models"][feature_name] = {
        "name": model_name,
        "batch_size": batch_size,
        "note": "cosine(summary+abstract) mapped to [0,1]"
    }

    meta["last_updated"] = datetime.datetime.now().isoformat(timespec="seconds")

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Metadata updated → {meta_path}")


# -------------------------------------------------------------------
# Arg parser
# -------------------------------------------------------------------
def parse_args():
    HERE = os.path.dirname(os.path.abspath(__file__))
    default_items = os.path.abspath(os.path.join(HERE, "..", "data_book", "items_filtered.tsv"))
    graph_dir = os.path.abspath(os.path.join(HERE, "..", "booksGraph"))

    ap = argparse.ArgumentParser(description="Add semantic similarity (summary+abstract) to book graph edges.")

    ap.add_argument("--items_tsv", default=default_items)
    ap.add_argument("--graph_in",  default=os.path.join(graph_dir, "graph_with_edges.pkl"))
    ap.add_argument("--graph_out", default=os.path.join(graph_dir, "graph_with_edges.pkl"))
    ap.add_argument("--meta_out",  default=os.path.join(graph_dir, "graph_with_edges_meta.json"))

    ap.add_argument("--feature_name", default="semantic_similarity")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=512)

    return ap.parse_args()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    args = parse_args()

    # Load graph
    G = load_graph(args.graph_in)
    print(f"[INFO] Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Build node texts
    nodes, texts = build_node_texts(G, args.items_tsv)
    print(f"[INFO] Loaded {len(texts)} text entries from items_filtered.tsv")

    # Compute embeddings
    emb = compute_embeddings(texts, args.model, args.batch_size)

    # Write similarity onto edges
    updated = write_semantic_to_edges(G, nodes, emb, args.feature_name)

    # Save graph_with_edges.pkl
    with open(args.graph_out, "wb") as f:
        pickle.dump(G, f)
    print(f"[DONE] Updated {updated:,} edges with '{args.feature_name}'")
    print(f"[INFO] Saved graph → {args.graph_out}")

    # Update metadata
    update_meta(
        args.meta_out,
        os.path.basename(args.graph_out),
        G,
        args.feature_name,
        updated,
        args.model,
        args.batch_size
    )


if __name__ == "__main__":
    main()
