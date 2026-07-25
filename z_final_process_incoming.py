import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import google.generativeai as genai
import json

# ----------------------------
# CONFIG
# ----------------------------
#GEMINI_API_KEY = "AIzaSyDtwkZ3Slp8zsMD5XZag0yXFlhJOXJJwu8"
GEMINI_API_KEY = "AIzaSyB4-ddj8HVaPFPcgoO0eNhZ_j2SVU-08XQ"
genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel("gemini-3-flash-preview")

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

def create_embedding(text):
    # This MUST match the model and dimensionality used in your migration script
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type="retrieval_query",
        output_dimensionality=768
    )
    return result['embedding']

# ----------------------------
# THE "FORCED METADATA" PROMPT
# ----------------------------
def generate_teacher_answer(json_context, question):
    prompt = f"""
SYSTEM ROLE: You are a Video Search Engine for the Sigma Web Development Course.
USER QUESTION: "{question}"

AVAILABLE VIDEO DATA:
{json_context}

INSTRUCTIONS:
You MUST extract the metadata from the JSON above. Your response must follow this EXACT structure:

📌 VIDEO SOURCE FOUND:
- Course: Sigma Web Development
- Lecture Number: [Insert 'lecture_number' from data]
- Video Title: [Insert 'video_title' from data]
- Timestamp: [Insert 'start_time'] to [Insert 'end_time']

📝 EXPLANATION:
[Provide a 2-3 sentence explanation of what is taught at this specific timestamp.]

🚀 ACTION:
"Go to Lecture #[Number] at [MM:SS] to learn about this topic."

(If no relevant data is in the JSON, say: "Topic not found in current course lectures.")
"""
    
    with open("prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    response = llm_model.generate_content(prompt)
    return response.text

# ----------------------------
# DATA PROCESSING
# ----------------------------
df = joblib.load("embeddings.joblib")
incoming_query = input("🔍 Search Course: ")

# 1. Vector Search
question_embedding = create_embedding(incoming_query)
embedding_matrix = np.vstack(df["embedding"].values)
similarities = cosine_similarity(embedding_matrix, [question_embedding]).flatten()

# 2. Get the #1 Best Match (To keep the answer focused on one specific timestamp)
best_idx = similarities.argsort()[-1]
row = df.iloc[best_idx]

# 3. Format strictly for the AI
formatted_match = {
    "video_title": str(row["title"]),
    "lecture_number": str(row["number"]),
    "start_time": format_time(row["start"]),
    "end_time": format_time(row["end"]),
    "text_content": str(row["text"])
}

json_context = json.dumps(formatted_match, indent=2)

# 4. Run
final_answer = generate_teacher_answer(json_context, incoming_query)

print("\n" + "="*30)
print(final_answer)
print("="*30)

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(final_answer)