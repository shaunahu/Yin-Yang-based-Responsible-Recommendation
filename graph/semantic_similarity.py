# semantic similarity analysis

# Import necessary libraries
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# Load your data
df = pd.read_csv('./data/news_item_subset_300.csv')

# Combine title and abstract
df['text'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode text into embeddings
embeddings = model.encode(df['text'].tolist(), convert_to_tensor=True)

# Store results
results = []

# Compute similarities
for i in tqdm(range(len(df))):
    for j in range(i + 1, len(df)):
        cosine_sim = util.cos_sim(embeddings[i], embeddings[j]).item()
        normalized_sim = (cosine_sim + 1) / 2  # Normalize into [0, 1]
        id_i = df.loc[i, 'item_id']
        id_j = df.loc[j, 'item_id']
        results.append(f"{id_i}\t{id_j}\t{normalized_sim:.4f}")

# Save to file
with open('./edge_data/semantic_similarity_titie_abstract_by_itemid.txt', 'w') as f:
    for line in results:
        f.write(line + '\n')

print("Done! Saved to semantic_similarity_by_itemid_normalized.txt")

