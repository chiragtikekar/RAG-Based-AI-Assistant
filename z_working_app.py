import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
import random
import json
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="AI Video Learning Platform", layout="wide")

# Custom CSS for UI
st.markdown("""
    <style>
    .review-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        border-right: 1px solid #eee;
        border-top: 1px solid #eee;
        border-bottom: 1px solid #eee;
        margin-bottom: 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .jump-btn {
        background-color: #ff4b4b;
        color: white !important;
        padding: 10px 20px;
        border-radius: 5px;
        text-decoration: none;
        display: inline-block;
        font-weight: bold;
        margin-top: 10px;
        transition: 0.3s;
    }
    .jump-btn:hover {
        background-color: #cc3333;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# NAVIGATION LOGIC (URL PARAMS)
# ----------------------------
query_params = st.query_params
url_lesson = query_params.get("lesson")
url_time = int(query_params.get("time", 0))

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_resource
def load_embeddings():
    return joblib.load("embeddings.joblib")

@st.cache_resource
def load_mcqs():
    with open("jsons/mcqs.json", "r") as f:
        return json.load(f)

df = load_embeddings()
mcq_data = load_mcqs()

# ----------------------------
# SIDEBAR NAVIGATION
# ----------------------------
lesson_map = {}
options_list = []
for number in sorted(df["number"].unique()):
    title = df[df["number"] == number]["title"].iloc[0]
    label = f"{number} - {title}"
    lesson_map[label] = number
    options_list.append(label)

default_index = 0
if url_lesson:
    # Try to match the URL lesson number to our list
    for i, label in enumerate(options_list):
        if label.split(" - ")[0] == str(url_lesson).zfill(2) or label.split(" - ")[0] == str(url_lesson):
            default_index = i
            break

selected_display = st.sidebar.radio("📚 Course Lessons", options_list, index=default_index)
selected_lesson_number = lesson_map[selected_display]

# ----------------------------
# VIDEO MAPPING (RESTORED & STRENGTHENED)
# ----------------------------
video_folder = "videos"
video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
selected_video = None

# Search for video using your original "#1" logic
for file in video_files:
    base_name = os.path.splitext(file)[0]
    first_part = base_name.split(" ")[0] # Gets "#1"
    file_num_cleaned = first_part.replace("#", "").zfill(2) # Gets "01"
    
    if file_num_cleaned == selected_lesson_number:
        selected_video = file
        break

video_path = os.path.join(video_folder, selected_video) if selected_video else None

# ----------------------------
# TABS
# ----------------------------
lesson_tab, quiz_tab = st.tabs(["🎥 Watch Lesson", "📝 Take Quiz"])

# ======================================================
# LESSON TAB
# ======================================================
with lesson_tab:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if video_path and os.path.exists(video_path):
            st.video(video_path, start_time=url_time)
            st.caption(f"Playing: {selected_video}")
        else:
            st.error(f"Video file not found for Lesson {selected_lesson_number}")
            st.info(f"Looked in: `{video_folder}` for a file starting with `#{int(selected_lesson_number)}`")
            
    with col2:
        st.subheader("🤖 Lesson Chatbot")
        user_query = st.text_input("Ask about this lecture:", placeholder="What is discussed here?")
        
        if st.button("Get Explanation", use_container_width=True):
            if user_query:
                def create_embedding(text_list):
                    r = requests.post("http://localhost:11434/api/embed", json={"model": "bge-m3", "input": text_list})
                    return r.json()["embeddings"]

                question_embedding = create_embedding([user_query])[0]
                lesson_chunks = df[df["number"] == selected_lesson_number]
                embedding_matrix = np.vstack(lesson_chunks["embedding"].values)
                similarities = cosine_similarity(embedding_matrix, [question_embedding]).flatten()
                top_indices = similarities.argsort()[::-1][:3]
                context = "\n\n".join(lesson_chunks.iloc[top_indices]["text"].values)
                st.markdown("### 📌 Insights")
                st.write(context)

# ======================================================
# QUIZ TAB
# ======================================================
with quiz_tab:
    # Logic to fetch MCQs for the current lesson
    lesson_mcqs = mcq_data.get(selected_lesson_number, [])

    if not lesson_mcqs:
        st.warning("No quiz questions found for this specific lesson.")
    else:
        if st.button("🚀 Start New Quiz"):
            st.session_state.quiz_list = random.sample(lesson_mcqs, min(5, len(lesson_mcqs)))
            st.session_state.user_answers = {}
            st.session_state.submitted = False

        if "quiz_list" in st.session_state:
            # We use a standard layout for better visibility
            for i, mcq in enumerate(st.session_state.quiz_list):
                st.markdown(f"#### Q{i+1}: {mcq['question']}")
                st.session_state.user_answers[i] = st.radio(
                    "Select Answer:", mcq["options"], index=None, key=f"q_{i}", label_visibility="collapsed"
                )
                st.write("") # Spacer

            if st.button("Submit My Answers"):
                st.session_state.submitted = True

            if st.session_state.get("submitted"):
                st.divider()
                score = 0
                
                # Grading loop
                for i, mcq in enumerate(st.session_state.quiz_list):
                    u_ans = st.session_state.user_answers.get(i)
                    correct_ans = mcq["options"][mcq["answer_index"]]
                    
                    if u_ans == correct_ans:
                        score += 1
                        st.success(f"✅ **Q{i+1}**: Correct!")
                    else:
                        m_s, s_s = int(mcq['start']) // 60, int(mcq['start']) % 60
                        # URL for the Jump Button
                        jump_url = f"/?lesson={mcq['number']}&time={int(mcq['start'])}"
                        
                        st.markdown(f"""
                        <div class="review-card">
                            <h4 style='color:#d32f2f; margin-top:0;'>❌ Question {i+1} Incorrect</h4>
                            <p><b>Your Answer:</b> {u_ans if u_ans else 'Skipped'}</p>
                            <p><b>Correct Answer:</b> <span style='color:green; font-weight:bold;'>{correct_ans}</span></p>
                            <hr style='margin:10px 0;'>
                            <p style='font-size: 0.9em; color: #555;'>Revise this at <b>{m_s:02d}:{s_s:02d}</b></p>
                            <a href="{jump_url}" target="_self" class="jump-btn">▶ Jump to Lecture Segment</a>
                        </div>
                        """, unsafe_allow_html=True)

                st.sidebar.metric("Last Score", f"{score}/{len(st.session_state.quiz_list)}")