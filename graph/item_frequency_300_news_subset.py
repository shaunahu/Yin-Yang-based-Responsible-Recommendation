import pandas as pd
import networkx as nx
import pickle
from collections import defaultdict
from tqdm import tqdm
import os


class FrequencyInjector:
    def __init__(self, impression_path, graph_input_path, graph_output_path, csv_output_path):
        self.impression_path = impression_path
        self.graph_input_path = graph_input_path
        self.graph_output_path = graph_output_path
        self.csv_output_path = csv_output_path
        self.impression_df = None
        self.G = None
        self.edge_frequency = defaultdict(int)

    def load_data(self):
        print("📥 Loading data...")
        self.impression_df = pd.read_csv(self.impression_path)
        with open(self.graph_input_path, "rb") as f:
            self.G = pickle.load(f)
        print(f"✅ Loaded graph with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.")

    def count_edge_frequencies(self):
        print("🔍 Counting edge frequencies from impression sequences...\n")
        for impression in tqdm(self.impression_df['impression'], desc="Processing impressions"):
            items = [x.split('-')[0] for x in str(impression).split()]
            item_index = {item: idx for idx, item in enumerate(items)}
            for u, v in self.G.edges():
                if u in item_index and v in item_index and item_index[u] < item_index[v]:
                    self.edge_frequency[(u, v)] += 1

    def inject_frequencies(self):
        print("📌 Injecting 'frequent' attribute to graph edges...")
        for u, v in self.G.edges():
            freq = self.edge_frequency.get((u, v), 0)
            self.G[u][v]['frequent'] = freq

        # Count non-zero frequency edges for logging
        nonzero_count = sum(1 for _, _, d in self.G.edges(data=True) if d.get('frequent', 0) > 0)
        print(f"🔎 Number of edges with frequency > 0: {nonzero_count}")

    def save_graph(self):
        with open(self.graph_output_path, "wb") as f:
            pickle.dump(self.G, f)
        print(f"✅ Updated graph saved to: {self.graph_output_path}")

    def export_nonzero_frequency_edges(self):
        print("📤 Exporting non-zero frequency edges to CSV...")
        edges = [
            (u, v, d['frequent']) for u, v, d in self.G.edges(data=True)
            if d.get('frequent', 0) > 0
        ]
        df = pd.DataFrame(edges, columns=["source", "target", "frequent"])
        df.to_csv(self.csv_output_path, index=False)
        print(f"✅ Exported to {self.csv_output_path}")

    def preview_top_edges(self, top_n=10):
        print(f"\n📈 Top {top_n} edges with highest 'frequent' value:\n")
        top_edges = sorted(
            [(u, v, d['frequent']) for u, v, d in self.G.edges(data=True)],
            key=lambda x: x[2], reverse=True
        )[:top_n]
        for u, v, freq in top_edges:
            print(f"{u} -> {v}, frequent: {freq}")

    def run_all(self):
        self.load_data()
        self.count_edge_frequencies()
        self.inject_frequencies()
        self.save_graph()
        self.export_nonzero_frequency_edges()
        self.preview_top_edges()


injector = FrequencyInjector(
    impression_path="./data/filtered_news_behaviors_300_subset.csv",
    graph_input_path="./graph_data/news_item_graph_full_augmented.pkl",
    graph_output_path="./graph_data/news_item_graph_full_augmented_with_frequency.pkl",
    csv_output_path="./output/nonzero_frequency_edges.csv"
)
injector.run_all()
