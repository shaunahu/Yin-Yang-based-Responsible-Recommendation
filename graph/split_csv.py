import pandas as pd

# Reload the TSV file to apply new filtering
df = pd.read_csv('./data/items.tsv', sep='\t')

# Extract needed columns: 0 (item_id), 1 (topic), 3 (title), 4 (abstract)
df_subset = df.iloc[:, [0, 1, 3, 4]]
df_subset.columns = ['item_id', 'topic', 'title', 'abstract']

# Define function to count words
def word_count(text):
    if pd.isna(text):
        return 0
    return len(str(text).split())

# Filter rows: both title and abstract must have >= 10 words
filtered_df = df_subset[
    (df_subset['title'].apply(word_count) >= 10) &
    (df_subset['abstract'].apply(word_count) >= 10)
]

# Select first 300 valid entries
final_df = filtered_df.head(300)

# Save to CSV
output_path = './data/news_item_subset_300.csv'
final_df.to_csv(output_path, index=False)

output_path
