# Normalize the third column from the uploaded similarity file
import pandas as pd

# Load the data
data = []
with open('./data_features/news_topic_similarity.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            data.append(parts)

# Convert to DataFrame
df = pd.DataFrame(data, columns=['item1', 'item2', 'similarity'])
df['similarity'] = df['similarity'].astype(float)

# Normalize similarity to [0,1] using the formula (similarity + 1) / 2
df['normalized_similarity'] = (df['similarity'] + 1) / 2

# Save to new TXT file
output_path = './data_features/news_topic_similarity_normalized.txt'
df[['item1', 'item2', 'normalized_similarity']].to_csv(
    output_path, sep='\t', index=False, header=False
)

output_path
