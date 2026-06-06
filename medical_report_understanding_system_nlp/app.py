import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
from reportlab.pdfgen import canvas
import tempfile

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Intelligent Medical Report Understanding System",
    page_icon="🩺",
    layout="wide"
)

MAX_LEN = 300

# =====================================================
# POSITIONAL ENCODING
# =====================================================

def positional_encoding(max_len, d_model):

    pos = np.arange(max_len)[:, np.newaxis]

    i = np.arange(d_model)[np.newaxis, :]

    angle_rates = 1 / np.power(
        10000,
        (2 * (i // 2)) / np.float32(d_model)
    )

    angle_rads = pos * angle_rates

    pe = np.zeros((max_len, d_model))

    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return pe.astype(np.float32)

# =====================================================
# CUSTOM POSITIONAL ENCODING LAYER
# =====================================================

class PositionalEncoding(Layer):

    def __init__(self, max_len=300, d_model=128, **kwargs):
        super().__init__(**kwargs)

        self.max_len = max_len
        self.d_model = d_model

        self.pos_encoding = tf.constant(
            positional_encoding(max_len, d_model)
        )

    def call(self, inputs):
        return inputs + self.pos_encoding[None, :, :]

    def get_config(self):
        config = super().get_config()

        config.update({
            "max_len": self.max_len,
            "d_model": self.d_model
        })

        return config

# =====================================================
# LOAD FILES
# =====================================================
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "medical_attention_model_fixed.keras"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

st.write("Current Directory:", BASE_DIR)
st.write("Model Exists:", MODEL_PATH.exists())
st.write("Tokenizer Exists:", TOKENIZER_PATH.exists())
st.write("Label Encoder Exists:", LABEL_ENCODER_PATH.exists())

# =====================================================
# LOAD FILES
# =====================================================

@st.cache_resource
def load_resources():

    try:

        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "PositionalEncoding": PositionalEncoding
            },
            compile=False,
            safe_mode=False
        )

        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)

        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)

        return model, tokenizer, label_encoder

    except Exception as e:
        st.exception(e)
        raise e


model, tokenizer, label_encoder = load_resources()
# =====================================================
# HEADER
# =====================================================

st.title("🩺 Intelligent Medical Report Understanding System")

st.markdown("""
Analyze medical reports and predict their specialty using
Transformer-based Healthcare NLP.
""")

# =====================================================
# INPUT SECTION
# =====================================================

report = st.text_area(
    "Enter Medical Report",
    height=250,
    placeholder="Paste medical report here..."
)

# =====================================================
# PREDICT
# =====================================================

if st.button("Analyze Report"):

    if report.strip() == "":
        st.warning("Please enter a medical report.")
        st.stop()

    sequence = tokenizer.texts_to_sequences([report])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(
        padded,
        verbose=0
    )

    pred_idx = np.argmax(prediction)

    specialty = label_encoder.inverse_transform(
        [pred_idx]
    )[0]

    confidence = float(
        np.max(prediction) * 100
    )

    # =================================================
    # RESULTS
    # =================================================

    st.header("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Specialty",
            specialty
        )

    with col2:
        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

    # =================================================
    # TOP 5 PREDICTIONS
    # =================================================

    st.subheader("Top 5 Predicted Specialties")

    probs = prediction[0]

    top5_idx = np.argsort(probs)[-5:][::-1]

    top5_labels = label_encoder.inverse_transform(
        top5_idx
    )

    top5_probs = probs[top5_idx] * 100

    top5_df = pd.DataFrame({
        "Specialty": top5_labels,
        "Probability (%)": np.round(top5_probs, 2)
    })

    st.dataframe(
        top5_df,
        use_container_width=True
    )

    # =================================================
    # DIAGNOSTIC TERMS
    # =================================================

    st.subheader("Diagnostic Importance Analysis")

    words = report.lower().split()

    keywords = [
        word for word in words
        if len(word) > 5
    ]

    keywords = list(dict.fromkeys(keywords))

    st.write(keywords[:15])

    # =================================================
    # POSITIONAL ENCODING HEATMAP
    # =================================================

    st.subheader("Positional Encoding Heatmap")

    pe = positional_encoding(
        100,
        64
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    img = ax.imshow(
        pe,
        aspect="auto"
    )

    ax.set_title(
        "Positional Encoding"
    )

    ax.set_xlabel(
        "Embedding Dimension"
    )

    ax.set_ylabel(
        "Token Position"
    )

    plt.colorbar(img)

    st.pyplot(fig)

    # =================================================
    # PDF REPORT
    # =================================================

    st.subheader("Download Medical Analysis Report")

    tmp_pdf = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    )

    c = canvas.Canvas(tmp_pdf.name)

    c.drawString(
        50,
        800,
        "Medical Analysis Report"
    )

    c.drawString(
        50,
        770,
        f"Predicted Specialty: {specialty}"
    )

    c.drawString(
        50,
        740,
        f"Confidence Score: {confidence:.2f}%"
    )

    c.drawString(
        50,
        710,
        "Important Terms:"
    )

    y = 680

    for word in keywords[:10]:

        c.drawString(
            70,
            y,
            word
        )

        y -= 20

    c.save()

    with open(tmp_pdf.name, "rb") as pdf_file:

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_file,
            file_name="medical_analysis_report.pdf",
            mime="application/pdf"
        )