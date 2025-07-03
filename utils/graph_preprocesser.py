import pandas as pd

from utils.utils import load_from_file, save_to_file
from common.constants import GRAPH_CLUSTER_FILE, TEST_USER_FILE


def get_cluster_from_graph():
    # create cluster set
    clusters = {}
    # create message set
    messages = set()

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
    return clusters, messages

def split_impression_column(df: pd.DataFrame, valid_ids: set) -> pd.DataFrame:
    """
    Filters and splits the 'impression' column into 'impression_1' and 'impression_0' columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame with a column 'impression' containing space-separated impression strings.
        valid_ids (set): A set of impression IDs (without -0 or -1 suffix) to filter.

    Returns:
        pd.DataFrame: Original DataFrame with 'impression_1' and 'impression_0' columns added.
    """
    def process_impressions(imp_str):
        items = imp_str.split()
        filtered = [item for item in items if item.split('-')[0] in valid_ids]
        group_0 = [item for item in filtered if item.endswith("-0")]
        group_1 = [item for item in filtered if item.endswith("-1")]
        return pd.Series([' '.join(group_0), ' '.join(group_1)])

    df[['impression_0', 'impression_1']] = df['impression'].apply(process_impressions)
    # remove rows with empty ['impression_0', 'impression_1']
    df = df[~((df['impression_0'] == '') & (df['impression_1'] == ''))]
    return df

if __name__ == "__main__":
    clusters, messages = get_cluster_from_graph()

    user_behavior = pd.read_csv(TEST_USER_FILE, sep="\t")
    users = split_impression_column(user_behavior, messages)
    save_to_file(users, "filtered_user_impressions.pkl")

