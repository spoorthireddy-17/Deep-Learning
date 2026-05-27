import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Mental Health Sentiment Monitor",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# CUSTOM CSS DESIGN
# =========================================================

st.markdown("""
<style>

.main {
    background: linear-gradient(
        135deg,
        #eef2ff,
        #dbeafe,
        #e0f2fe
    );
    color: #111827;
}

.stApp {
    background: linear-gradient(
        135deg,
        #eef2ff,
        #dbeafe,
        #e0f2fe
    );
}

h1, h2, h3, h4 {
    color: #1e293b;
}

p, label, div {
    color: #334155;
}

textarea {
    background-color: #ffffff !important;
    color: #111827 !important;
    border-radius: 12px !important;
    border: 1px solid #cbd5e1 !important;
}

.stButton > button {

    background: linear-gradient(
        90deg,
        #60a5fa,
        #818cf8
    );

    color: white;

    border-radius: 12px;

    border: none;

    padding: 12px 30px;

    font-size: 18px;

    font-weight: bold;
}

.stButton > button:hover {

    background: linear-gradient(
        90deg,
        #3b82f6,
        #6366f1
    );

    color: white;
}

.metric-card {

    background-color: rgba(255,255,255,0.8);

    padding: 20px;

    border-radius: 16px;

    margin-top: 10px;

    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
}

.guidance-box {

    background-color: rgba(255,255,255,0.75);

    padding: 20px;

    border-radius: 16px;

    margin-top: 10px;

    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL FILES
# =========================================================

model = load_model("mental_health_rnn_model.h5")

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    encoder = pickle.load(file)


# =========================================================
# PARAMETERS
# =========================================================

max_length = 50


# =========================================================
# HEADER SECTION
# =========================================================

st.markdown("""
<h1 style='text-align:center;'>
🧠 AI-Based Mental Health Sentiment Monitoring System
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<h4 style='text-align:center; color:#cbd5e1;'>
Emotion Detection using Simple Recurrent Neural Networks
</h4>
""", unsafe_allow_html=True)

st.markdown("---")


# =========================================================
# ABOUT PROJECT
# =========================================================

st.header("📘 About the Project")

st.write("""
This AI-powered application analyzes emotional sentiment from user text using Natural Language Processing (NLP) and Simple Recurrent Neural Networks (RNN).

### Importance of Emotional AI
Emotional AI helps identify emotional patterns, monitor mental wellness, and support early intervention.

### NLP Applications
- Emotion Detection
- Mental Health Monitoring
- AI Chatbots
- Social Media Sentiment Analysis
- Counseling Assistance Systems

### Role of RNN
RNNs process words sequentially and remember previous words using hidden states, helping understand emotional context.
""")

st.markdown("---")


# =========================================================
# USER INPUT AREA
# =========================================================

st.header("✍️ Express Your Feelings")

st.write("### Sample Sentences")

col1, col2 = st.columns(2)

with col1:
    st.info("I feel very lonely these days")
    st.info("Nobody understands my feelings")

with col2:
    st.info("I am excited and happy today")
    st.info("I feel anxious about my future")


user_input = st.text_area(
    "User Text",
    placeholder="Enter your thoughts or feelings here...",
    height=180
)


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_emotion(text):

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=max_length,
        padding='post',
        truncating='post'
    )

    prediction = model.predict(padded)

    predicted_class = np.argmax(prediction)

    emotion = encoder.inverse_transform(
        [predicted_class]
    )[0]

    confidence = np.max(prediction) * 100

    return emotion, confidence, prediction[0]


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button("🔍 Analyze Emotion"):

    if user_input.strip() == "":

        st.warning("Please enter some text.")

    else:

        emotion, confidence, probabilities = predict_emotion(user_input)

        st.markdown("---")


        # =========================================================
        # OUTPUT SECTION
        # =========================================================

        st.header("📊 Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class='metric-card'>
            <h3>Emotion</h3>
            <h2>{emotion}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='metric-card'>
            <h3>Confidence</h3>
            <h2>{confidence:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)

        with col3:

            if emotion.lower() in [
                'depression',
                'anxiety',
                'suicidal',
                'stress'
            ]:

                status = "Needs Attention"

            else:

                status = "Emotionally Stable"

            st.markdown(f"""
            <div class='metric-card'>
            <h3>Status</h3>
            <h2>{status}</h2>
            </div>
            """, unsafe_allow_html=True)


        # =========================================================
        # VISUALIZATION
        # =========================================================

        st.markdown("---")

        st.header("📈 Emotion Confidence Graph")

        class_labels = encoder.classes_

        fig, ax = plt.subplots(figsize=(9,4))

        bars = ax.bar(
            class_labels,
            probabilities
        )

        ax.set_xlabel("Emotion Category")

        ax.set_ylabel("Confidence")

        ax.set_title("Sentiment Probability Distribution")

        plt.xticks(rotation=30)

        st.pyplot(fig)


        # =========================================================
        # WELLNESS GUIDANCE
        # =========================================================

        st.markdown("---")

        st.header("💡 Emotional Wellness Guidance")

        if emotion.lower() in ['depression', 'suicidal']:

            st.markdown("""
            <div class='guidance-box'>

            🌿 Take a short break and talk with someone you trust.

            🌿 Avoid staying isolated.

            🌿 Practice mindfulness and breathing exercises.

            🌿 Consider reaching out to a mental health professional.

            </div>
            """, unsafe_allow_html=True)

        elif emotion.lower() == 'anxiety':

            st.markdown("""
            <div class='guidance-box'>

            🌿 Listen to calming music.

            🌿 Focus on one task at a time.

            🌿 Try meditation or a short walk.

            🌿 Take deep breaths slowly.

            </div>
            """, unsafe_allow_html=True)

        elif emotion.lower() in ['joy', 'happy', 'normal']:

            st.markdown("""
            <div class='guidance-box'>

            🌟 Keep maintaining healthy habits.

            🌟 Share positivity with others.

            🌟 Continue self-care and balanced routines.

            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class='guidance-box'>

            🌿 Stay connected with supportive people.

            🌿 Maintain healthy sleep habits.

            🌿 Spend time doing activities you enjoy.

            </div>
            """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "AI-Based Mental Health Sentiment Monitoring System using SimpleRNN"
)