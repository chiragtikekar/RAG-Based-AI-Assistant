import joblib
import random
import re

# Load chunks
df = joblib.load("embeddings.joblib")

def clean_sentence(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def generate_mcq_from_chunk(chunk_text):
    sentences = chunk_text.split(".")
    sentences = [clean_sentence(s) for s in sentences if len(s.split()) > 5]

    if not sentences:
        return None

    # Pick main sentence
    correct_sentence = sentences[0]

    # Create question
    question = f"What does the following statement describe?\n\n\"{correct_sentence}\""

    # Create distractors from other random chunks
    other_chunks = df.sample(4)["text"].values
    distractors = []

    for oc in other_chunks:
        oc_sentences = oc.split(".")
        oc_sentences = [clean_sentence(s) for s in oc_sentences if len(s.split()) > 5]
        if oc_sentences:
            distractors.append(oc_sentences[0])

    options = distractors[:3]
    options.append(correct_sentence)

    random.shuffle(options)

    correct_option = options.index(correct_sentence)

    return {
        "question": question,
        "options": options,
        "answer_index": correct_option
    }

# Example usage
sample_chunk = df.iloc[10]["text"]
mcq = generate_mcq_from_chunk(sample_chunk)

print("Q:", mcq["question"])
for i, opt in enumerate(mcq["options"]):
    print(f"{chr(65+i)}) {opt}")

print("Correct:", chr(65 + mcq["answer_index"]))