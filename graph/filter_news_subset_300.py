import pandas as pd


behaviors_path = "./data/new_behaviors.tsv"
news_items_path = "./data/news_item_subset_300.csv"


behaviors_df = pd.read_csv(behaviors_path, sep='\t', header=None)
news_items_df = pd.read_csv(news_items_path)


behaviors_df.columns = ['user_id', 'time', 'history', 'impression']


valid_news_ids = set(news_items_df['item_id'].astype(str))


def contains_valid_news(history, impression, valid_ids):
    history_ids = history.split() if pd.notna(history) else []
    impression_ids = [x.split('-')[0] for x in impression.split()] if pd.notna(impression) else []
    return bool(set(history_ids + impression_ids) & valid_ids)


filtered_behaviors_df = behaviors_df[
    behaviors_df.apply(lambda row: contains_valid_news(row['history'], row['impression'], valid_news_ids), axis=1)
]

# Save
output_path = "./data/filtered_news_behaviors_300_subset.csv"
filtered_behaviors_df.to_csv(output_path, index=False)

filtered_behaviors_df.to_csv(output_path, index=False)
print(f"Filtered data has been saved: {output_path}")