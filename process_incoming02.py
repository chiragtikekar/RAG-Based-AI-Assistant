import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import google.generativeai as genai

# ----------------------------
# CONFIG
# ----------------------------
#GEMINI_API_KEY = "AIzaSyDtwkZ3Slp8zsMD5XZag0yXFlhJOXJJwu8"
GEMINI_API_KEY = "AIzaSyB4-ddj8HVaPFPcgoO0eNhZ_j2SVU-08XQ"
genai.configure(api_key=GEMINI_API_KEY)

llm_model = genai.GenerativeModel("gemini-3-flash-preview")

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
    Context:
    {context}

    Question: {user_query}

    Answer briefly using only the context.
    If not found say: Answer not found in video.
    """

    print("\n--- DEBUG: PROMPT SENT TO LLM ---")
    print(prompt)
    print("---------------------------------\n")

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
top_indices = similarities.argsort()[::-1]

selected_chunks = []
selected_texts = []

for idx in top_indices:
    text = lesson_chunks.iloc[idx]["text"]

    # avoid very similar chunks
    if all(text not in t for t in selected_texts):
        selected_chunks.append(idx)
        selected_texts.append(text)

    if len(selected_chunks) == 2:
        break

context = " ".join(selected_texts)


# Build Context & Final Answer
context = "\n".join(retrieved_df["text"].values)
final_answer = generate_answer(context, incoming_query)

print("\n💡 Answer:\n", final_answer.strip())