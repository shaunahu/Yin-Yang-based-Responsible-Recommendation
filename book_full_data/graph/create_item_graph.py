#!/usr/bin/env python3
# read_graph.py
# Preview edges from booksGraph/{graph.pkl | graph_with_edges.pkl}

import os
import pickle
import argparse
import itertools


def parse_args():
    ap = argparse.ArgumentParser(
        description="Preview edges from a BOOKS graph pickle"
    )
    ap.add_argument(
        "--graph",
        default="../booksGraph/graph_with_edges.pkl",
        help="Path to graph pickle "
             "(default: ../booksGraph/graph_with_edges.pkl)",
    )
    ap.add_argument(
        "-n",
        type=int,
        default=10,
        help="Number of edges to preview (default: 10)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    graph_path = os.path.abspath(args.graph)

    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph not found: {graph_path}")

    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    print(f"[INFO] Graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"[INFO] Previewing first {args.n} edges:\n")

    for idx, (u, v, data) in enumerate(itertools.islice(G.edges(data=True), args.n), start=1):
        print(f"{idx}. {u} -- {v}")
        if data:
            for k, val in data.items():
                print(f"      {k}: {val}")
        else:
            print("      (no attributes)")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
