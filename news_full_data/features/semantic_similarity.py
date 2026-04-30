#!/usr/bin/env python3
# semantic_similarity.py
# Sep 19,2025
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


# ---------------------------- helpers ----------------------------

def load_graph(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph not found: {path}")
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G


def build_node_texts(
    G,
    items_csv: str,
    prefer_node_attrs: bool = True
) -> Tuple[List[str], List[str]]:
    """
    For each node in the graph (G.nodes() order), produce (nodes, texts),
    where texts = "title abstract".
    If node doesn't have title/abstract, look up from items_filtered.csv.
    """
    nodes = list(G.nodes())
    texts: List[str] = []

    # Check attrs present on nodes
    has_title_attr = prefer_node_attrs and any("title" in G.nodes[n] for n in nodes)
    has_abs_attr   = prefer_node_attrs and any("abstract" in G.nodes[n] for n in nodes)

    # Prepare CSV fallback if needed
    id2row = {}
    titles: List[str] = []
    abstracts: List[str] = []
    if not (has_title_attr and has_abs_attr):
        if not os.path.exists(items_csv):
            raise FileNotFoundError(f"Items file not found: {items_csv}")
        df = pd.read_csv(items_csv, dtype=str)
        if "item_id" not in df.columns:
            raise ValueError(f"{items_csv} must contain 'item_id' column")
        id2row = {iid: idx for idx, iid in enumerate(df["item_id"].tolist())}
        titles = df.get("title", pd.Series([""] * len(df))).fillna("").tolist()
        abstracts = df.get("abstract", pd.Series([""] * len(df))).fillna("").tolist()

    missing = 0
    for n in nodes:
        title = ""
        abstract = ""
        if has_title_attr and "title" in G.nodes[n]:
            title = G.nodes[n].get("title") or ""
        if has_abs_attr and "abstract" in G.nodes[n]:
            abstract = G.nodes[n].get("abstract") or ""

        if (title == "" and abstract == "") and id2row:
            if n in id2row:
                r = id2row[n]
                title = titles[r]
                abstract = abstracts[r]
            else:
                missing += 1  # item not in CSV; leave empty

        texts.append(f"{title} {abstract}".strip())

    if missing > 0:
        print(f"[WARN] {missing} nodes not found in items_filtered.csv; used empty text.")
    return nodes, texts


def compute_node_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int
) -> np.ndarray:
    """
    Encode texts to unit-normalized embeddings (cosine == dot).
    Returns float32 array of shape (n, d).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading model: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)

    print("[INFO] Encoding node texts → embeddings (unit vectors)")
    with torch.inference_mode():
        emb = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            device=device,
            normalize_embeddings=True
        )  # (n, d) float32
    return emb.detach().to("cpu").numpy().astype(np.float32, copy=False)


def write_semantic_on_existing_edges(
    G,
    nodes: List[str],
    emb: np.ndarray,
    feature_name: str
) -> int:
    """
    For each existing edge (u,v) in G, compute cosine similarity from node embeddings,
    map to [0,1], and set as edge attribute feature_name. No new edges are added.
    Returns number of edges updated.
    """
    node2idx: Dict[str, int] = {nid: i for i, nid in enumerate(nodes)}
    updated = 0

    edges = list(G.edges())
    print(f"[INFO] Updating '{feature_name}' for {len(edges):,} existing edges...")
    for u, v in tqdm(edges, desc="semantic to edges"):
        iu = node2idx.get(u)
        iv = node2idx.get(v)
        if iu is None or iv is None:
            continue
        sim = float(np.dot(emb[iu], emb[iv]))  # cosine (unit vectors)
        sim01 = (sim + 1.0) * 0.5              # map [-1,1] -> [0,1]
        G[u][v][feature_name] = sim01
        updated += 1
    return updated


def update_meta(
    meta_path: str,
    graph_filename: str,
    G,
    feature_name: str,
    updated_count: int,
    model_name: str,
    batch_size: int
):
    """Append/merge feature info into graph meta JSON."""
    import datetime

    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    # Ensure core fields
    meta.setdefault("graph_file", graph_filename)
    meta["nodes"] = G.number_of_nodes()
    meta["edges"] = G.number_of_edges()

    # Edge feature list
    edge_feats = meta.get("edge_features", [])
    if feature_name not in edge_feats:
        edge_feats.append(feature_name)
    meta["edge_features"] = edge_feats

    # Per-feature counts
    efc = meta.get("edge_feature_counts", {})
    efc[feature_name] = int(updated_count)
    meta["edge_feature_counts"] = efc

    # Model info
    meta.setdefault("models", {})
    meta["models"][feature_name] = {
        "name": model_name,
        "batch_size": batch_size,
        "note": "cosine(title+abstract) rescaled to [0,1]"
    }

    # Timestamp
    meta["last_updated"] = datetime.datetime.now().isoformat(timespec="seconds")

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Metadata updated -> {meta_path}")


# ---------------------------- main ----------------------------

def parse_args():
    HERE = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.abspath(os.path.join(HERE, "..", "data_news", "items_filtered.csv"))
    default_graph_dir = os.path.abspath(os.path.join(HERE, "..", "newsGraph"))

    ap = argparse.ArgumentParser(description="Add semantic (title+abstract) similarity to existing edges in a NetworkX graph.")
    ap.add_argument("--items_csv", default=default_data, help="Path to items_filtered.csv")
    ap.add_argument("--graph_in",  default=os.path.join(default_graph_dir, "graph_with_edges.pkl"),
                    help="Input graph (with nodes+edges). Default: newsGraph/graph_with_edges.pkl")
    ap.add_argument("--graph_out", default=os.path.join(default_graph_dir, "graph_with_edges.pkl"),
                    help="Output graph path (default overwrites input)")
    ap.add_argument("--meta_out",  default=os.path.join(default_graph_dir, "graph_with_edges_meta.json"),
                    help="Metadata JSON path to update")
    ap.add_argument("--feature_name", default="semantic_similarity",
                    help="Edge attribute name to write (default: semantic_similarity)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Sentence-Transformer model (default: all-MiniLM-L6-v2)")
    ap.add_argument("--batch_size", type=int, default=512, help="Embedding batch size (default: 512)")
    ap.add_argument("--prefer_node_attrs", action="store_true",
                    help="Prefer existing node attributes (title/abstract) over CSV when available")
    return ap.parse_args()


def main():
    args = parse_args()

    # 1) Load graph
    G = load_graph(args.graph_in)
    print(f"[INFO] Loaded graph: nodes={G.number_of_nodes():,}, edges={G.number_of_edges():,}")

    # 2) Prepare texts for nodes (prefer node attrs if requested)
    nodes, texts = build_node_texts(G, args.items_csv, prefer_node_attrs=args.prefer_node_attrs)

    # 3) Compute node embeddings
    emb = compute_node_embeddings(texts, args.model, args.batch_size)  # (n, d)

    # 4) Write semantic similarity to existing edges
    updated = write_semantic_on_existing_edges(G, nodes, emb, args.feature_name)

    # 5) Save graph
    with open(args.graph_out, "wb") as f:
        pickle.dump(G, f)
    print(f"[DONE] Wrote '{args.feature_name}' on {updated:,} edges.")
    print(f"[INFO] Saved graph -> {args.graph_out}")

    # 6) Update metadata JSON
    update_meta(
        meta_path=args.meta_out,
        graph_filename=os.path.basename(args.graph_out),
        G=G,
        feature_name=args.feature_name,
        updated_count=updated,
        model_name=args.model,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
