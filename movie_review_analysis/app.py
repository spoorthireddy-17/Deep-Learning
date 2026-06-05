import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'[^\w\s]', '', text)

    text = " ".join([
        word for word in text.split()
        if word not in stop_words
    ])

    return text
# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    rnn = tf.keras.models.load_model("simple_rnn_model.keras")
    lstm = tf.keras.models.load_model("lstm_model.keras")
    gru = tf.keras.models.load_model("gru_model.keras")
    return rnn, lstm, gru

simple_rnn_model, lstm_model, gru_model = load_models()

# -----------------------------
# Load Tokenizer
# -----------------------------
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

tokenizer = load_tokenizer()
st.sidebar.write(
    f"Vocabulary Size: {len(tokenizer.word_index)}"
)

MAX_LEN = 200

# -----------------------------
# Prediction Function
# -----------------------------
import re

def preprocess_text(text):
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    return text


def predict_sentiment(model, review):

    # Apply same preprocessing
    clean_review = preprocess_text(review)

    st.write("Original Review:")
    st.write(review)

    st.write("Cleaned Review:")
    st.write(clean_review)

    clean_review = preprocess_text(review)

    seq = tokenizer.texts_to_sequences([clean_review])
    st.write("Token Sequence:")
    st.write(seq)

    st.write("Sequence Length:")
    st.write(len(seq[0]))

    padded = pad_sequences(
    seq,
    maxlen=200,
    padding='pre',
    truncating='pre'
    )

    raw = model.predict(
        padded,
        verbose=0
    )

    st.write("Raw Prediction:")
    st.write(raw)

    prob = float(raw[0][0])

    positive_prob = prob
    negative_prob = 1 - prob

    sentiment = (
        "Positive"
        if prob >= 0.5
        else "Negative"
    )

    confidence = (
        max(
            positive_prob,
            negative_prob
        ) * 100
    )

    return (
        sentiment,
        confidence,
        positive_prob,
        negative_prob
    )
# -----------------------------
# Header
# -----------------------------
st.title("🎬 Movie Review Sentiment Analysis System")

st.markdown(
    "### Deep Learning Based Sentiment Classification"
)

st.markdown("---")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Model Selection")

selected_model = st.sidebar.radio(
    "Choose Model",
    ["SimpleRNN", "LSTM", "GRU"]
)

# -----------------------------
# Input Area
# -----------------------------
review = st.text_area(
    "Enter your movie review here...",
    height=200,
    placeholder="Type your movie review..."
)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")
        st.stop()

    model_map = {
        "SimpleRNN": simple_rnn_model,
        "LSTM": lstm_model,
        "GRU": gru_model
    }

    selected = model_map[selected_model]

    sentiment, confidence, pos_prob, neg_prob = predict_sentiment(
        selected,
        review
    )

    # -----------------------------
    # Output Area
    # -----------------------------
    st.subheader("Prediction Result")

    if sentiment == "Positive":
        st.success(f"Sentiment: {sentiment}")
    else:
        st.error(f"Sentiment: {sentiment}")

    st.info(f"Confidence: {confidence:.2f}%")

    st.markdown("---")

    # -----------------------------
    # Probability Display
    # -----------------------------
    st.subheader("Probability Distribution")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Positive Probability",
            f"{pos_prob*100:.2f}%"
        )

    with col2:
        st.metric(
            "Negative Probability",
            f"{neg_prob*100:.2f}%"
        )

    # -----------------------------
    # Confidence Chart
    # -----------------------------
    st.subheader("Confidence Chart")

    prob_df = pd.DataFrame({
        "Sentiment": ["Positive", "Negative"],
        "Probability": [
            pos_prob * 100,
            neg_prob * 100
        ]
    })

    st.bar_chart(
        prob_df.set_index("Sentiment")
    )

    st.markdown("---")

    # -----------------------------
    # Compare All Models
    # -----------------------------
    st.subheader(
        "Comparison of All Models on the Same Review"
    )

    comparison = []

    models = {
        "SimpleRNN": simple_rnn_model,
        "LSTM": lstm_model,
        "GRU": gru_model
    }

    for name, model in models.items():

        sent, conf, pos, neg = predict_sentiment(
            model,
            review
        )

        comparison.append({
            "Model": name,
            "Prediction": sent,
            "Confidence (%)": round(conf, 2),
            "Positive Probability (%)": round(pos*100, 2),
            "Negative Probability (%)": round(neg*100, 2)
        })

    comparison_df = pd.DataFrame(comparison)

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    st.subheader("Model Confidence Comparison")

    chart_df = comparison_df[
        ["Model", "Confidence (%)"]
    ]

    st.bar_chart(
        chart_df.set_index("Model")
    )

st.markdown("---")
st.caption(
    "Movie Review Sentiment Analysis using SimpleRNN, LSTM, and GRU"
)