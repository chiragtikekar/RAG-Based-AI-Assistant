import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import google.generativeai as genai

# ----------------------------
# CONFIG
# ----------------------------
GEMINI_API_KEY = "AIzaSyDtwkZ3Slp8zsMD5XZag0yXFlhJOXJJwu8"
genai.configure(api_key=GEMINI_API_KEY)

llm_model = genai.GenerativeModel("gemini-1.5-flash")

# ----------------------------
# UPDATED EMBEDDING FUNCTION
# ----------------------------
def create_embedding(text):
    # This MUST match the model and dimensionality used in your migration script
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type="retrieval_query",
        output_dimensionality=768
    )
    return result['embedding']

def generate_answer(context, question):
    prompt = f"""
    You are an AI tutor helping a student understand a programming lecture.
    Use ONLY the information provided in the context below.
    If the answer is not present in the context, reply exactly: "Answer not found in video."

    Context: {context}
    Student Question: {question}
    Helpful Explanation:
    """
    response = llm_model.generate_content(prompt)
    return response.text

# ----------------------------
# LOAD & RUN
# ----------------------------
df = joblib.load("embeddings.joblib")
incoming_query = input("Ask a Question: ")

# Get Gemini Embedding for the query
question_embedding = create_embedding(incoming_query)

# Similarity Search
embedding_matrix = np.vstack(df["embedding"].values)
similarities = cosine_similarity(embedding_matrix, [question_embedding]).flatten()

# Retrieval
top_k = 3
top_indices = similarities.argsort()[::-1][:top_k]
retrieved_df = df.iloc[top_indices]

# Build Context & Final Answer
context = "\n".join(retrieved_df["text"].values)
final_answer = generate_answer(context, incoming_query)

print("\n💡 Answer:\n", final_answer.strip())