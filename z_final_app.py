import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import random
import json
import google.generativeai as genai 
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="AI Video Learning Platform", layout="wide")

# Gemini Configuration
#GEMINI_API_KEY = "AIzaSyDtwkZ3Slp8zsMD5XZag0yXFlhJOXJJwu8"
GEMINI_API_KEY = "AIzaSyB4-ddj8HVaPFPcgoO0eNhZ_j2SVU-08XQ"
genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel("gemini-3-flash-preview")

# Custom CSS for UI
st.markdown("""
    <style>
    .review-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
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
        background-color: #e04343;
        text-decoration: none;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_resource
def load_data():
    df = joblib.load("embeddings.joblib")
    with open("jsons/mcqs.json", "r") as f:
        mcqs = json.load(f)
    return df, mcqs

try:
    df, mcq_data = load_data()
except Exception as e:
    st.error(f"Error loading data files: {e}")
    st.stop()

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
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
# NAVIGATION & URL LOGIC
# ----------------------------
# Get current URL state
query_params = st.query_params
url_lesson = query_params.get("lesson")
url_time = int(query_params.get("time", 0))

# Map lessons for sidebar
lesson_map = {}
options_list = []
for number in sorted(df["number"].unique()):
    title = df[df["number"] == number]["title"].iloc[0]
    label = f"{number} - {title}"
    lesson_map[label] = number
    options_list.append(label)

# Determine sidebar index based on URL
default_idx = 0
if url_lesson:
    for i, label in enumerate(options_list):
        if label.split(" - ")[0].zfill(2) == str(url_lesson).zfill(2):
            default_idx = i
            break

selected_display = st.sidebar.radio("📚 Course Lessons", options_list, index=default_idx)
selected_lesson_number = lesson_map[selected_display]

# ----------------------------
# VIDEO MAPPING
# ----------------------------
video_folder = "videos"
video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
selected_video_file = None

for file in video_files:
    base_name = os.path.splitext(file)[0]
    file_num_cleaned = base_name.split(" ")[0].replace("#", "").zfill(2)
    if file_num_cleaned == selected_lesson_number:
        selected_video_file = file
        break

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
        if selected_video_file:
            video_path = os.path.join(video_folder, selected_video_file)
            # Use the time from URL for initial start
            st.video(video_path, start_time=url_time)
            st.caption(f"Playing: {selected_video_file} at {format_time(url_time)}")
        else:
            st.error("Lecture video not found.")
            
    with col2:
        st.subheader("🤖 HTML Tutor")
        user_query = st.text_input("Ask about this lecture:", placeholder="e.g., What are tags?")
        
        if st.button("Get Explanation", use_container_width=True):
            if user_query:
                with st.spinner("Analyzing course content..."):
                    # 1. Similarity Search
                    q_emb = create_embedding(user_query)
                    emb_matrix = np.vstack(df["embedding"].values)
                    sims = cosine_similarity(emb_matrix, [q_emb]).flatten()
                    
                    row = df.iloc[sims.argsort()[-1]]
                    
                    # 2. Generate AI Answer
                    prompt = f"""
                    SYSTEM: You are a helpful tutor for an HTML course.
                    CONTEXT FROM VIDEO: Lesson {row['number']}, Title: {row['title']}
                    TRANSCRIPT SEGMENT: {row['text']}
                    USER QUESTION: {user_query}
                    
                    INSTRUCTION: Answer briefly based on the transcript. 
                    Mention that more details are in Lecture {row['number']} at {format_time(row['start'])}.
                    """
                    response = llm_model.generate_content(prompt)

                    # 3. Store match for the persistent UI button
                    st.session_state.last_response = response.text
                    st.session_state.match_lesson = row['number']
                    st.session_state.match_time = int(row['start'])
                    st.session_state.match_label = format_time(row['start'])

        # Display persistent result if it exists
        if "last_response" in st.session_state:
            st.markdown("---")
            st.write(st.session_state.last_response)
            
            jump_url = f"/?lesson={st.session_state.match_lesson}&time={st.session_state.match_time}"
            st.markdown(f"""
                <a href="{jump_url}" target="_self" class="jump-btn">
                    ▶ Open Lecture {st.session_state.match_lesson} at {st.session_state.match_label}
                </a>
            """, unsafe_allow_html=True)

# ======================================================
# QUIZ TAB
# ======================================================
with quiz_tab:
    # Option to toggle between Current Lesson and All Lessons
    quiz_mode = st.radio("Quiz Mode:", ["Current Lesson Only", "Global (All Lessons)"], horizontal=True)

    if quiz_mode == "Current Lesson Only":
        lesson_mcqs = mcq_data.get(selected_lesson_number, [])
    else:
        # Flatten all MCQs from all lessons into one list
        lesson_mcqs = [item for sublist in mcq_data.values() for item in sublist]

    if not lesson_mcqs:
        st.info("No quiz available for this selection yet.")
    else:
        if st.button("🚀 Start New Quiz"):
            # Randomly pick 5 questions from the available pool
            st.session_state.quiz_list = random.sample(lesson_mcqs, min(5, len(lesson_mcqs)))
            st.session_state.user_answers = {}
            st.session_state.quiz_submitted = False

        if "quiz_list" in st.session_state:
            for i, mcq in enumerate(st.session_state.quiz_list):
                st.markdown(f"#### Q{i+1}: {mcq['question']}")
                # If it's a Global Quiz, show which lesson this question is from (optional hint)
                if quiz_mode == "Global (All Lessons)":
                    st.caption(f"Topic from Lecture {mcq['number']}")
                
                st.session_state.user_answers[i] = st.radio(
                    f"Options_{i}", mcq["options"], index=None, key=f"q_{i}", label_visibility="collapsed"
                )

            if st.button("Submit Quiz"):
                st.session_state.quiz_submitted = True

            if st.session_state.get("quiz_submitted"):
                st.divider()
                score = 0
                for i, mcq in enumerate(st.session_state.quiz_list):
                    u_ans = st.session_state.user_answers.get(i)
                    correct_ans = mcq["options"][mcq["answer_index"]]
                    
                    if u_ans == correct_ans:
                        score += 1
                        st.success(f"✅ Q{i+1}: Correct!")
                    else:
                        q_jump_url = f"/?lesson={mcq['number']}&time={int(mcq['start'])}"
                        
                        # Added color: #333 to ensure text is visible on the white card
                        st.markdown(f"""
                        <div class="review-card">
                            <h4 style='color:#d32f2f; margin-top:0;'>❌ Question {i+1} Incorrect</h4>
                            <p style='color: #333;'><b>Correct Answer:</b> <span style='color: #2e7d32;'>{correct_ans}</span></p>
                            <hr style='border: 0.5px solid #eee;'>
                            <p style='font-size: 0.9em; color: #555;'>Watch the explanation in <b>Lecture {mcq['number']}</b> at <b>{format_time(mcq['start'])}</b></p>
                            <a href="{q_jump_url}" target="_self" class="jump-btn">
                                ▶ Jump to Lecture {mcq['number']} Segment
                            </a>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.sidebar.metric("Latest Quiz Score", f"{score}/{len(st.session_state.quiz_list)}")