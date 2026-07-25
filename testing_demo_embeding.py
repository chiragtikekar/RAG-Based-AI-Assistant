import pandas as pd
import joblib
import google.generativeai as genai
from tqdm import tqdm

# --- CONFIG ---
GEMINI_API_KEY = "AIzaSyDtwkZ3Slp8zsMD5XZag0yXFlhJOXJJwu8"
genai.configure(api_key=GEMINI_API_KEY)

# Use the new mainline model
EMBEDDING_MODEL = "models/gemini-embedding-001"

def re_embed_data():
    # 1. Load your old data
    print("🔄 Loading existing embeddings...")
    df = joblib.load("embeddings.joblib")
    
    # Extract only the text
    texts = df['text'].tolist()
    
    print(f"🚀 Generating {len(texts)} new Gemini embeddings...")
    new_embeddings = []
    
    # Process in batches
    batch_size = 50 
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        try:
            # Task type must be 'retrieval_document' for the stored database
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch,
                task_type="retrieval_document",
                output_dimensionality=768 # Forces 768 dimensions (matches old models)
            )
            new_embeddings.extend(result['embedding'])
        except Exception as e:
            print(f"\n❌ Error at batch {i}: {e}")
            return # Stop if API fails so we don't save empty data

    # 3. Update DataFrame and Save
    if len(new_embeddings) == len(df):
        df['embedding'] = new_embeddings
        joblib.dump(df, "embeddings.joblib")
        print("\n✅ Successfully updated embeddings.joblib with Gemini vectors!")
    else:
        print("\n⚠️ Failed to generate all embeddings. File not saved.")

if __name__ == "__main__":
    re_embed_data()