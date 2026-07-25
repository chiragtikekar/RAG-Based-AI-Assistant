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
st.title("🎓 AI Video Learning Platform")

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
# BUILD LESSON MAP FROM DATAFRAME
# ----------------------------
lesson_map = {}
for number in sorted(df["number"].unique()):
    title = df[df["number"] == number]["title"].iloc[0]
    lesson_map[f"{number} - {title}"] = number

st.sidebar.title("📚 Course Lessons")
selected_display = st.sidebar.radio(
    "Select Lesson",
    list(lesson_map.keys()),
    key="selected_lesson_number"
)
selected_lesson_number = lesson_map[selected_display]

# Reset quiz if lesson changes
if "active_lesson" not in st.session_state:
    st.session_state.active_lesson = selected_lesson_number

if st.session_state.active_lesson != selected_lesson_number:
    st.session_state.active_lesson = selected_lesson_number
    if "current_quiz" in st.session_state:
        del st.session_state.current_quiz

# ----------------------------
# MAP VIDEO TO LESSON NUMBER (ROBUST VERSION)
# ----------------------------
video_folder = "videos"
video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

selected_video = None

for file in video_files:
    base_name = os.path.splitext(file)[0]      # remove .mp4
    first_part = base_name.split(" ")[0]       # "#1"
    file_number = first_part.replace("#", "").zfill(2)

    if file_number == selected_lesson_number:
        selected_video = file
        break

if selected_video:
    video_path = os.path.join(video_folder, selected_video)
else:
    video_path = None
# ----------------------------
# TABS
# ----------------------------
lesson_tab, quiz_tab = st.tabs(["🎥 Lesson", "📝 Quiz"])

# ======================================================
# LESSON TAB
# ======================================================
with lesson_tab:

    if video_path:
        st.video(video_path)
    else:
        st.warning("No video file found for this lesson.")

    st.subheader("❓ Ask a Doubt About This Lesson")
    user_query = st.text_input("Type your question here")

    if st.button("Get Explanation"):

        def create_embedding(text_list):
            r = requests.post(
                "http://localhost:11434/api/embed",
                json={"model": "bge-m3", "input": text_list}
            )
            return r.json()["embeddings"]

        question_embedding = create_embedding([user_query])[0]

        lesson_chunks = df[df["number"] == selected_lesson_number]

        embedding_matrix = np.vstack(lesson_chunks["embedding"].values)

        similarities = cosine_similarity(
            embedding_matrix,
            [question_embedding]
        ).flatten()

        top_indices = similarities.argsort()[::-1][:3]
        retrieved_df = lesson_chunks.iloc[top_indices]

        context = "\n\n".join(retrieved_df["text"].values)

        st.subheader("📌 Explanation")
        st.write(context)
# ======================================================
# QUIZ TAB
# ======================================================
with quiz_tab:

    st.subheader("📝 Lesson Quiz")

    if selected_lesson_number not in mcq_data:
        st.warning("No MCQs available for this lesson.")
        st.stop()

    lesson_mcqs = []

    for lecture in mcq_data:
        lesson_mcqs.extend(mcq_data[lecture])

    # ----------------------------
    # GENERATE QUIZ BUTTON
    # ----------------------------
    if st.button("Generate Quiz"):
        st.session_state.current_quiz = random.sample(
            lesson_mcqs,
            min(5, len(lesson_mcqs))
        )
        st.session_state.answers = {}

    # ----------------------------
    # SHOW QUIZ
    # ----------------------------
    if "current_quiz" in st.session_state:

        for i, mcq in enumerate(st.session_state.current_quiz):

            st.markdown(f"### Q{i+1}. {mcq['question']}")

            selected_option = st.radio(
                "Choose an answer:",
                mcq["options"],
                index=None,
                key=f"quiz_{i}"
            )

            st.session_state.answers[i] = selected_option

        # ----------------------------
        # SUBMIT BUTTON
        # ----------------------------
        if st.button("Submit Quiz"):

            score = 0
            wrong_questions = []

            for i, mcq in enumerate(st.session_state.current_quiz):

                user_answer = st.session_state.answers.get(i)

                if user_answer is None:
                    continue

                if mcq["options"].index(user_answer) == mcq["answer_index"]:
                    score += 1
                else:
                    wrong_questions.append(mcq)

            st.success(f"Your Score: {score} / {len(st.session_state.current_quiz)}")

            # ----------------------------
            # REVISION SUGGESTION
            # ----------------------------
            if wrong_questions:

                st.error("📍 Review these questions and revise the video:")

                for i, wrong in enumerate(wrong_questions):

                    minutes_start = int(wrong["start"]) // 60
                    seconds_start = int(wrong["start"]) % 60

                    minutes_end = int(wrong["end"]) // 60
                    seconds_end = int(wrong["end"]) % 60

                    correct_answer = wrong["options"][wrong["answer_index"]]

                    st.markdown(
                        f"""
                ❌ **Incorrect Question**

                **Question:** {wrong["question"]}

                ✅ **Correct Answer:** {correct_answer}

                📚 **Lecture:** {wrong["number"]} - {wrong["title"]}

                🔎 **Revise from:**  
                {minutes_start:02d}:{seconds_start:02d} → {minutes_end:02d}:{seconds_end:02d}
                """
                    )

                    # 🔥 GO TO LECTURE BUTTON
                    if st.button(f"▶ Go to Lecture {wrong['number']}", key=f"go_{i}"):

                        st.session_state.selected_lesson_number = f"{wrong['number']} - {wrong['title']}"
                        st.rerun()