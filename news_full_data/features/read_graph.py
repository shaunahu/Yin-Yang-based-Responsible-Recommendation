#!/usr/bin/env python3
# preview_graph_edges.py
# Show the first N edges and their features from graph_with_edges.pkl

import os
import pickle
import argparse
import itertools

def parse_args():
    ap = argparse.ArgumentParser(description="Preview edges from graph_with_edges.pkl")
    ap.add_argument("--graph", default="../newsGraph/graph_with_edges.pkl",
                    help="Path to graph_with_edges.pkl")
    ap.add_argument("-n", type=int, default=101,
                    help="Number of edges to preview (default: 10)")
    return ap.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.graph):
        raise FileNotFoundError(f"Graph not found: {args.graph}")

    with open(args.graph, "rb") as f:
        G = pickle.load(f)

    print(f"[INFO] Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"[INFO] Previewing first {args.n} edges:\n")

    for idx, (u, v, data) in enumerate(itertools.islice(G.edges(data=True), args.n)):
        print(f"{idx+1}. {u} -- {v}")
        if data:
            for k, v in data.items():
                print(f"     {k}: {v}")
        else:
            print("     (no features)")
    print("\n[DONE]")

if __name__ == "__main__":
    main()
