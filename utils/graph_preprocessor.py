import pandas as pd
import numpy as np

from utils.utils import load_from_file, save_to_file
from common.constants import GRAPH_CLUSTER_FILE, TEST_USER_FILE, TEST_FILTERED_USER_FILE

POSITIVE_WEIGHT = 1.0
NEGATIVE_WEIGHT = -0.0
ALPHA = 0.01

def get_cluster_from_graph():
    # create cluster set
    clusters = {}
    # create message set
    messages = set()
    message_cluster_map = {}

    # read graph
    graph_file = GRAPH_CLUSTER_FILE
    G = load_from_file(graph_file)
    for node, attr in G.nodes(data=True):
        messages.add(node)

        cluster_id = attr["cluster"]
        if cluster_id not in clusters:
            clusters[cluster_id] = set()
        else:
            clusters[cluster_id].add(node)
        message_cluster_map[node] = cluster_id
    return clusters, messages, message_cluster_map

def split_impression_column(df: pd.DataFrame, valid_ids: set) -> pd.DataFrame:
    """
    Filters and splits the 'impression' column into 'impression_1' and 'impression_0' columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame with a column 'impression' containing space-separated impression strings.
        valid_ids (set): A set of impression IDs (without -0 or -1 suffix) to filter.

    Returns:
        pd.DataFrame: Original DataFrame with 'impression_1' and 'impression_0' columns added. Remove rows with empty 'impression_1' and 'impression_0'
    """
    def process_impressions(imp_str):
        items = imp_str.split()
        filtered = [item for item in items if item.split('-')[0] in valid_ids]
        group_0 = [item.split("-0")[0] for item in filtered if item.endswith("-0")]
        group_1 = [item.split("-1")[0] for item in filtered if item.endswith("-1")]
        return pd.Series([group_0, group_1])

    df[['impression_0', 'impression_1']] = df['impression'].apply(process_impressions)
    # remove rows with empty ['impression_0', 'impression_1']
    df = df[~((df['impression_0'].str.len() == 0) & (df['impression_1'].str.len() == 0))].reset_index(drop=True)
    return df


def main():
    clusters, messages, message_cluster_map = get_cluster_from_graph()

    user_behavior = pd.read_csv(TEST_USER_FILE, sep="\t")
    users = split_impression_column(user_behavior, messages)
    impressions = users[["impression_1", "impression_0"]]
    num_clusters = len(set(clusters))
    user_beliefs = np.zeros((len(users), num_clusters))

    belief_list = []

    for idx, row in impressions.iterrows():
        cluster_scores = np.zeros(num_clusters)

        # Positive evidence from clicked items
        for msg_id in row["impression_1"]:
            cluster_id = message_cluster_map.get(msg_id)
            if cluster_id is not None:
                cluster_idx = cluster_id - 1
                cluster_scores[cluster_idx] += POSITIVE_WEIGHT

        # Negative evidence from non-clicked items
        for msg_id in row["impression_0"]:
            cluster_id = message_cluster_map.get(msg_id)
            if cluster_id is not None:
                cluster_idx = cluster_id - 1
                cluster_scores[cluster_idx] += NEGATIVE_WEIGHT

        # Apply smoothing
        smoothed = cluster_scores + ALPHA
        smoothed = np.clip(smoothed, 1e-6, None)

        # Normalize to get belief distribution
        belief = smoothed / smoothed.sum()
        user_beliefs[idx] = belief
        belief_list.append(belief)

    # Add belief vector column to the users DataFrame
    users['belief'] = belief_list

    # calculate avg belief
    avg_belief_distribution = np.mean(user_beliefs, axis=0)
    print(f"Average belief distribution: {avg_belief_distribution}")

    users.to_csv("utils/users_with_belief.csv", index=False)


if __name__ == "__main__":
    main()
