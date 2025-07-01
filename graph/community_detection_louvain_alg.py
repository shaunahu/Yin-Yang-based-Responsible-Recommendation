# Define a reusable class to encapsulate the community detection and analysis process
import pickle
import pandas as pd
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import seaborn as sns


# Define a backward-compatible version of CommunityAnalyzer with colormap fix
class CommunityAnalyzerCompat:
    def __init__(self, graph_path, node_info_path):
        with open(graph_path, 'rb') as f:
            self.G = pickle.load(f)
        self.node_info_path = node_info_path
        self.node_topic = {}
        self.node_community = {}
        self.communities = []
        self.central_nodes = {}
        self.community_topic_df = None
        self.output_dir = "./community_outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    def compute_combined_weight(self):
        for u, v, data in self.G.edges(data=True):
            values = [
                data.get('frequency'),
                data.get('semantic_similarity'),
                data.get('sentiment_similarity'),
                data.get('topic_similarity'),
            ]
            valid_values = [v for v in values if v is not None]
            data['combined_weight'] = sum(valid_values) / len(valid_values) if valid_values else 0.0

    def detect_communities(self):
        from networkx.algorithms.community import louvain_communities
        self.communities = louvain_communities(self.G, weight='combined_weight', resolution=1.0)
        self.node_community = {node: i for i, comm in enumerate(self.communities) for node in comm}

    def visualize_communities(self):
        pos = nx.spring_layout(self.G, seed=42)
        colors = [self.node_community.get(node, -1) for node in self.G.nodes()]
        plt.figure(figsize=(12, 10))
        nx.draw_networkx_nodes(self.G, pos, node_color=colors, cmap=plt.get_cmap("Set3"), node_size=40)
        nx.draw_networkx_edges(self.G, pos, alpha=0.2)
        plt.title("Louvain Communities Visualization (combined weight)")
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, "full_graph_communities.png"))
        plt.close()

    def compute_central_nodes(self):
        self.central_nodes = {}
        for i, community in enumerate(self.communities):
            subgraph = self.G.subgraph(community)
            central_node = max(subgraph.degree, key=lambda x: x[1])[0]
            self.central_nodes[i] = central_node

    def compute_topic_distribution(self):
        df = pd.read_csv(self.node_info_path)
        self.node_topic = dict(zip(df['item_id'].astype(str), df['topic'].astype(str).str.strip()))

        topic_dist = defaultdict(lambda: defaultdict(int))
        for node, comm_id in self.node_community.items():
            topic = self.node_topic.get(node)
            if topic:
                topic_dist[comm_id][topic] += 1

        self.community_topic_df = pd.DataFrame.from_dict(topic_dist, orient='index').fillna(0).astype(int)
        self.community_topic_df.to_csv(os.path.join(self.output_dir, "community_topic_distribution.csv"))

    def plot_topic_distributions(self):
        if self.community_topic_df is None:
            return

        # Normalize for heatmap
        topic_dist_norm = self.community_topic_df.div(self.community_topic_df.sum(axis=1), axis=0)

        # Stacked Bar Plot
        bar_path = os.path.join(self.output_dir, "topic_distribution_stacked_bar.png")
        self.community_topic_df.plot(kind="bar", stacked=True, figsize=(12, 6), cmap=plt.get_cmap("tab20"))
        plt.title("Community Topic Distribution (Stacked Bar)")
        plt.xlabel("Community ID")
        plt.ylabel("Topic Count")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.savefig(bar_path)
        plt.close()

        # Heatmap
        heatmap_path = os.path.join(self.output_dir, "topic_distribution_heatmap.png")
        plt.figure(figsize=(12, 6))
        sns.heatmap(topic_dist_norm, cmap="YlGnBu", annot=True, fmt=".2f", cbar=True)
        plt.title("Community Topic Distribution (Heatmap - Proportions)")
        plt.xlabel("Topic")
        plt.ylabel("Community ID")
        plt.tight_layout()
        plt.savefig(heatmap_path)
        plt.close()

    def save_node_community_csv(self):
        df = pd.DataFrame(list(self.node_community.items()), columns=["node", "community"])
        df.to_csv(os.path.join(self.output_dir, "node_community_assignments.csv"), index=False)

    def save_edge_summary_csv(self):
        edge_data = []
        for u, v, data in self.G.edges(data=True):
            edge_data.append({
                "node_u": u,
                "node_v": v,
                "community_u": self.node_community.get(u),
                "community_v": self.node_community.get(v),
                "frequency": data.get("frequency"),
                "semantic_similarity": data.get("semantic_similarity"),
                "sentiment_similarity": data.get("sentiment_similarity"),
                "topic_similarity": data.get("topic_similarity"),
                "combined_weight": data.get("combined_weight")
            })
        pd.DataFrame(edge_data).to_csv(os.path.join(self.output_dir, "edge_weights_with_communities.csv"), index=False)

    def save_individual_community_plots(self):
        for comm_id, nodes in enumerate(self.communities):
            subgraph = self.G.subgraph(nodes)
            pos = nx.spring_layout(subgraph, seed=42)
            plt.figure(figsize=(8, 6))
            nx.draw_networkx_nodes(subgraph, pos, node_color="skyblue", node_size=60)
            nx.draw_networkx_edges(subgraph, pos, alpha=0.3)
            plt.title(f"Community {comm_id}")
            plt.axis('off')
            path = os.path.join(self.output_dir, f"community_{comm_id}.png")
            plt.savefig(path)
            plt.close()

    def run_all(self):
        self.compute_combined_weight()
        self.detect_communities()
        self.visualize_communities()
        self.compute_central_nodes()
        self.compute_topic_distribution()
        self.plot_topic_distributions()
        self.save_node_community_csv()
        self.save_edge_summary_csv()
        self.save_individual_community_plots()

analyzer = CommunityAnalyzerCompat(
    graph_path="./graph_data/news_item_graph_full_augmented_with_frequency.pkl",
    node_info_path="./data/news_item_subset_300.csv"
)
analyzer.run_all()