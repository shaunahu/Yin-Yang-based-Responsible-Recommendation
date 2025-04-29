import pickle
import pandas as pd
import networkx as nx
from tqdm import tqdm

class GraphAugmentor:
    def __init__(self, graph_path):
        with open(graph_path, 'rb') as f:
            self.G = pickle.load(f)

    def load_similarity_files(self, semantic_path, sentiment_path):
        self.semantic_df = pd.read_csv(semantic_path, sep='\t', header=None, names=['item1', 'item2', 'semantic_similarity'])
        self.sentiment_df = pd.read_csv(sentiment_path, sep='\t', header=None, names=['item1', 'item2', 'sentiment_similarity'])

        self.semantic_dict = {}
        for _, row in self.semantic_df.iterrows():
            self.semantic_dict[(row['item1'], row['item2'])] = row['semantic_similarity']
            self.semantic_dict[(row['item2'], row['item1'])] = row['semantic_similarity']

        self.sentiment_dict = {}
        for _, row in self.sentiment_df.iterrows():
            self.sentiment_dict[(row['item1'], row['item2'])] = row['sentiment_similarity']
            self.sentiment_dict[(row['item2'], row['item1'])] = row['sentiment_similarity']

    def add_semantic_and_sentiment_similarity(self):
        for u, v in tqdm(self.G.edges(), desc='Adding semantic and sentiment similarity'):
            semantic_sim = self.semantic_dict.get((u, v), None)
            sentiment_sim = self.sentiment_dict.get((u, v), None)
            self.G[u][v]['semantic_similarity'] = semantic_sim
            self.G[u][v]['sentiment_similarity'] = sentiment_sim

    def load_topic_info(self, node_info_path, topic_similarity_path):
        df = pd.read_csv(node_info_path)
        self.node_topic = dict(zip(df['item_id'], df['topic']))

        self.topic_sim_df = pd.read_csv(topic_similarity_path, sep='\t', header=None, names=['topic1', 'topic2', 'topic_similarity'])
        self.topic_sim_dict = {}
        for _, row in self.topic_sim_df.iterrows():
            self.topic_sim_dict[(row['topic1'], row['topic2'])] = row['topic_similarity']
            self.topic_sim_dict[(row['topic2'], row['topic1'])] = row['topic_similarity']

    def add_topic_similarity(self):
        for u, v in tqdm(self.G.edges(), desc='Adding topic similarity'):
            topic_u = self.node_topic.get(u, None)
            topic_v = self.node_topic.get(v, None)
            if topic_u and topic_v:
                topic_sim = self.topic_sim_dict.get((topic_u, topic_v), None)
            else:
                topic_sim = None
            self.G[u][v]['topic_similarity'] = topic_sim

    def save_graph(self, output_path):
        with open(output_path, 'wb') as f:
            pickle.dump(self.G, f)

    def get_graph(self):
        return self.G

# Example usage:
augmentor = GraphAugmentor('./graph_data/news_item_subset_300_graph.pkl')
augmentor.load_similarity_files('./edge_data/semantic_similarity_titie_abstract_by_itemid.txt', './edge_data/sentiment_similarity_titile_abstract_by_itemid.txt')
augmentor.add_semantic_and_sentiment_similarity()
augmentor.load_topic_info('./edge_data/news_item_subset_300.csv', './edge_data/news_topic_similarity_normalized.txt')
augmentor.add_topic_similarity()
augmentor.save_graph('./graph_data/news_item_graph_full_augmented.pkl')

# To read back
with open('./graph_data/news_item_graph_full_augmented.pkl', 'rb') as f:
    G = pickle.load(f)
print(list(G.edges(data=True))[:5])  # 查看前5条边及其属性
