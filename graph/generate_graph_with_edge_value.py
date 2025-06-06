import pickle
import pandas as pd
import networkx as nx
from tqdm import tqdm

def clean_topic_similarity_file(input_path, output_path):
    """
    Clean topic similarity file
    """
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            parts = line.strip().split()  # support space or tab
            if len(parts) == 3:
                topic1, topic2, similarity = parts[0].strip(), parts[1].strip(), parts[2].strip()
                fout.write(f"{topic1}\t{topic2}\t{similarity}\n")

class GraphAugmentor:
    def __init__(self, graph_path):
        with open(graph_path, 'rb') as f:
            self.G = pickle.load(f)

    def load_similarity_files(self, semantic_path, sentiment_path):
        self.semantic_df = pd.read_csv(semantic_path, sep='\t', header=None, names=['item1', 'item2', 'semantic_similarity'])
        self.sentiment_df = pd.read_csv(sentiment_path, sep='\t', header=None, names=['item1', 'item2', 'sentiment_similarity'])

        self.semantic_dict = {}
        for _, row in self.semantic_df.iterrows():
            item1, item2 = row['item1'].strip(), row['item2'].strip()
            sim = row['semantic_similarity']
            self.semantic_dict[(item1, item2)] = sim
            self.semantic_dict[(item2, item1)] = sim

        self.sentiment_dict = {}
        for _, row in self.sentiment_df.iterrows():
            item1, item2 = row['item1'].strip(), row['item2'].strip()
            sim = row['sentiment_similarity']
            self.sentiment_dict[(item1, item2)] = sim
            self.sentiment_dict[(item2, item1)] = sim

    def add_semantic_and_sentiment_similarity(self):
        for u, v in tqdm(self.G.edges(), desc='Adding semantic and sentiment similarity'):
            semantic_sim = self.semantic_dict.get((u, v), None)
            sentiment_sim = self.sentiment_dict.get((u, v), None)
            self.G[u][v]['semantic_similarity'] = semantic_sim
            self.G[u][v]['sentiment_similarity'] = sentiment_sim

    def load_topic_info(self, node_info_path, topic_similarity_path):
        df = pd.read_csv(node_info_path)
        self.node_topic = dict(zip(df['item_id'].astype(str), df['topic'].astype(str).str.strip()))

        self.topic_sim_df = pd.read_csv(topic_similarity_path, sep='\t', header=None, names=['topic1', 'topic2', 'topic_similarity'])
        self.topic_sim_dict = {}
        for _, row in self.topic_sim_df.iterrows():
            t1, t2 = row['topic1'].strip(), row['topic2'].strip()
            sim = row['topic_similarity']
            self.topic_sim_dict[(t1, t2)] = sim
            self.topic_sim_dict[(t2, t1)] = sim

    def add_topic_similarity(self):
        for u, v in tqdm(self.G.edges(), desc='Adding topic similarity'):
            topic_u = self.node_topic.get(u, None)
            topic_v = self.node_topic.get(v, None)
            if topic_u and topic_v:
                if topic_u == topic_v:
                    topic_sim = 1.0
                else:
                    topic_sim = self.topic_sim_dict.get((topic_u, topic_v), None)
                    if topic_sim is None:
                        print(f"[WARN] Missing topic similarity for: ({topic_u}, {topic_v})")
            else:
                topic_sim = None
                print(f"[WARN] Missing topic for node {u} or {v}")
            self.G[u][v]['topic_similarity'] = topic_sim

    def save_graph(self, output_path):
        with open(output_path, 'wb') as f:
            pickle.dump(self.G, f)

    def get_graph(self):
        return self.G

# === main ===

# Step 1: clean topic similarity txt file
clean_topic_similarity_file(
    input_path='./edge_data/news_topic_similarity_normalized.txt',
    output_path='./edge_data/news_topic_similarity_cleaned.txt'
)

# Step 2: Use GraphAugmentor 
augmentor = GraphAugmentor('./graph_data/news_item_subset_300_graph.pkl')
augmentor.load_similarity_files(
    './edge_data/semantic_similarity_titie_abstract_by_itemid.txt',
    './edge_data/sentiment_similarity_titile_abstract_by_itemid.txt'
)
augmentor.add_semantic_and_sentiment_similarity()
augmentor.load_topic_info(
    node_info_path='./edge_data/news_item_subset_300.csv',
    topic_similarity_path='./edge_data/news_topic_similarity_cleaned.txt'
)
augmentor.add_topic_similarity()
# save graph
augmentor.save_graph('./graph_data/news_item_graph_full_augmented.pkl')

# print some sample node with edge information
print("\n=== Example edges with attributes ===")
G_aug = augmentor.get_graph()
count = 0
for u, v, data in G_aug.edges(data=True):
    print(f"Edge: ({u}, {v})")
    print(f"  semantic_similarity: {data.get('semantic_similarity')}")
    print(f"  sentiment_similarity: {data.get('sentiment_similarity')}")
    print(f"  topic_similarity: {data.get('topic_similarity')}")
    print("-----")
    count += 1
    if count >= 5:
        break