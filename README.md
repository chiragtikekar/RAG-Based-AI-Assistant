# 🎓 RAG-Based AI Teaching Assistant

An AI-powered educational assistant that enables students to ask questions from recorded lecture videos using Retrieval-Augmented Generation (RAG).

Instead of relying on a generic Large Language Model, the assistant retrieves relevant lecture content using semantic search and generates context-aware responses with Gemini API.

---

## Features

- 🎥 Upload lecture videos
- 🎙 Automatic speech-to-text transcription using Faster-Whisper
- ✂ Transcript chunking and preprocessing
- 🧠 Semantic search using vector embeddings
- 🤖 Context-aware question answering with Gemini API
- ⏱ Timestamp-based navigation to lecture videos
- 📝 AI-generated MCQs from lecture content
- 📊 Interactive Streamlit interface

---

## System Architecture

```
Lecture Video
      │
      ▼
Extract Audio (FFmpeg)
      │
      ▼
Speech-to-Text
(Faster-Whisper)
      │
      ▼
Transcript Cleaning
      │
      ▼
Chunking
      │
      ▼
Vector Embeddings
      │
      ▼
Semantic Search
(Cosine Similarity)
      │
      ▼
Relevant Context
      │
      ▼
Gemini API
      │
      ▼
AI Response + Timestamp
```

---

## Technologies

- Python
- Streamlit
- Gemini API
- Faster-Whisper
- Pandas
- NumPy
- Scikit-learn
- FFmpeg
- JSON

---

## Project Workflow

1. Upload lecture video
2. Extract audio from video
3. Transcribe lecture using Faster-Whisper
4. Clean and preprocess transcript
5. Split transcript into semantic chunks
6. Generate embeddings
7. Perform semantic similarity search
8. Retrieve relevant lecture context
9. Generate response using Gemini API
10. Display timestamp-linked answer

---

## Folder Structure

```
📂 rag-ai-teaching-assistant
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
├── transcripts/
├── embeddings/
├── videos/
│
├── utils/
│   ├── preprocessing.py
│   ├── embedding.py
│   ├── retrieval.py
│   └── llm.py
│
└── assets/
```

---

## Installation

```bash
git clone https://github.com/yourusername/rag-ai-teaching-assistant.git

cd rag-ai-teaching-assistant

pip install -r requirements.txt

streamlit run app.py
```

---

## Future Improvements

- Vector database integration (FAISS/ChromaDB)
- Multi-document retrieval
- Azure OpenAI support
- Lecture summarization
- Multi-language transcription
- Student learning analytics

---

## Author

**Chirag Tikekar**

LinkedIn:
https://linkedin.com/in/chiragtikekar

GitHub:
https://github.com/chiragtikekar
