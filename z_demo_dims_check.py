import joblib
import numpy as np

# Load your file
df = joblib.load("embeddings.joblib")

# Get the first embedding from the 'embedding' column
first_embedding = df["embedding"].iloc[0]

# Check the length
print(f"Vector type: {type(first_embedding)}")
print(f"Dimension: {len(first_embedding)}")

# If you want to see the shape of the whole matrix
embedding_matrix = np.vstack(df["embedding"].values)
print(f"Matrix shape: {embedding_matrix.shape}")