# Sentiment similarity between titles and abstracts of each item
# (titles-titles + abstracts-abstracts) Sentiment similarity/2 (Average)

# def generate_sentiment(self):
#     blob = TextBlob(self.title + " " + self.abstract)
#     self.normalized_sentiment = (blob.sentiment.polarity + 1) / 2

# Import required libraries
import pandas as pd
from textblob import TextBlob
from tqdm import tqdm

# Reload the CSV file
df = pd.read_csv('./data/news_item_subset_300.csv')

# Combine 'title' and 'abstract' into one text field
df['text'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')

# Initialize list to hold the results
results = []

# Use tqdm for a progress bar
for i in tqdm(range(len(df))):
    blob_i = TextBlob(df.loc[i, 'text'])
    sentiment_i = (blob_i.sentiment.polarity + 1) / 2  # normalized to [0,1]
    id_i = df.loc[i, 'item_id']  # Get item_id
    
    for j in range(i + 1, len(df)):
        blob_j = TextBlob(df.loc[j, 'text'])
        sentiment_j = (blob_j.sentiment.polarity + 1) / 2
        id_j = df.loc[j, 'item_id']
        
        # Calculate sentiment difference
        sentiment_diff = abs(sentiment_i - sentiment_j)
        
        # Save using item_ids instead of indices
        results.append(f"{id_i}\t{id_j}\t{sentiment_diff:.4f}")

# Save all results to a new txt file
output_path = './edge_data/sentiment_similarity_by_itemid.txt'
with open(output_path, 'w') as f:
    for line in results:
        f.write(line + '\n')

print(output_path)
