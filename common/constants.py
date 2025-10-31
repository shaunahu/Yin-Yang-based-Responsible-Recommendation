SELECTIVE_METHODS = ["LightGCN", "DGCF", "NGCF", "SGL", "ENMF", "DiffRec", "NCL", "LDiffRec"]
SELECTIVE_DATASETS = ["movie", "book", "news"]
ITEM_FILE = "items.tsv"
ITEM_PICKLE_FILE = "items.pkl"
USER_PICKLE_FILE = "users.pkl"
USER_FILE = "new_behaviors.tsv"
USER_BELIF_FILE = "user_behaviors_belief.tsv"
MAX_WORDS = 300
DIM = 100
NUM_THREADS = 32

GRAPH_CLUSTER_FILE = "graph/graph_data/news_item_graph_with_clusters.pkl"
TEST_USER_FILE = "resource/news/new_behaviors.tsv"