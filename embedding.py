from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model and your existing data
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the embeddings CSV
df = pd.read_csv("embeddings.csv")

# Extract the sentence column
sentences = df["sentence"].tolist()

# Extract only the embedding values (exclude the 'sentence' column)
embeddings = df.drop(columns=["sentence"]).values

# New user query
query = "How can Medicare help me?"

# Embed the query
query_embedding = model.encode([query])

# Compute cosine similarity
similarities = cosine_similarity(query_embedding, embeddings)[0]

# Get index of most similar sentence
best_match_idx = np.argmax(similarities)

# Print result
print(f"Query: {query}")
print(f"Most similar FAQ: {sentences[best_match_idx]}")
print(f"Similarity score: {similarities[best_match_idx]:.4f}")