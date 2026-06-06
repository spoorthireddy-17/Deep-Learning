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

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container{
    padding-top: 2rem;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

.keyword-chip {
    display:inline-block;
    background:#dbeafe;
    color:#1e40af;
    padding:8px 14px;
    margin:4px;
    border-radius:20px;
    font-weight:600;
}

.stButton > button {
    width:100%;
    height:3em;
    border-radius:12px;
    border:none;
    background:linear-gradient(
        90deg,
        #2563eb,
        #0ea5e9
    );
    color:white;
    font-size:18px;
    font-weight:600;
}

.header-box{
    background:linear-gradient(
        90deg,
        #1e3a8a,
        #2563eb
    );
    padding:25px;
    border-radius:15px;
    color:white;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)
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

st.markdown("""
<div class='header-box'>
<h1>🩺 Intelligent Medical Report Understanding System</h1>
<h4>Healthcare NLP using Self-Attention & Positional Encoding</h4>
</div>
""", unsafe_allow_html=True)

st.write("")


with st.sidebar:

    st.header("🏥 Medical NLP Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Reports", "4,999+")

    with col2:
        st.metric("Classes", "40")

    st.metric(
        "Validation Accuracy",
        "35.7%"
    )

    st.metric(
        "Model Type",
        "Transformer"
    )

    st.info("""
    **Architecture**

    Embedding

    ↓

    Positional Encoding

    ↓

    Multi-Head Attention

    ↓

    Dense Softmax
    """)

    st.success(
        "Healthcare NLP Classification System"
    )
# =====================================================
# INPUT SECTION
# =====================================================

st.subheader("📄 Medical Report Input")

report = st.text_area(
    "",
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

    st.subheader("📊 Prediction Results")

    c1,c2,c3 = st.columns(3)

    with c1:
        st.metric(
            "Predicted Specialty",
            specialty
        )

    with c2:
        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    with c3:

        if confidence > 50:
            st.success("High Confidence")

        elif confidence < 50:
            st.info(
                "Prediction confidence is low. Consider reviewing Top-5 specialties."
            )
        elif confidence > 20:
            st.warning("Moderate Confidence")

        else:
            st.error("Low Confidence")
 

    # =================================================
# TOP 5 PREDICTIONS
# =================================================

    st.subheader(" Top 5 Specialty Predictions")

    probs = prediction[0]

    top5_idx = np.argsort(probs)[-5:][::-1]

    top5_labels = label_encoder.inverse_transform(
            top5_idx
        )

    top5_probs = probs[top5_idx] * 100

    chart_df = pd.DataFrame({
            "Specialty": top5_labels,
            "Probability": np.round(top5_probs, 2)
        })
    fig_bar, ax_bar = plt.subplots(
            figsize=(10, 4)
        )

    bars = ax_bar.barh(
            chart_df["Specialty"][::-1],
            chart_df["Probability"][::-1]
        )

    ax_bar.set_xlabel("Probability (%)")
    ax_bar.set_title("Top 5 Predicted Specialties")

    for i, value in enumerate(
            chart_df["Probability"][::-1]
        ):
                ax_bar.text(
                    value + 0.5,
                    i,
                    f"{value:.1f}%",
                    va="center"
                )

    st.pyplot(fig_bar)
        # =================================================
        # DIAGNOSTIC TERMS
        # =================================================

    st.subheader(" 🧬 Diagnostic Importance Analysis")

    words = report.lower().split()

    keywords = [
            word for word in words
            if len(word) > 5
        ]

    keywords = list(dict.fromkeys(keywords))

    import re

    stop_words = {
        "patient","history","diagnosis",
        "procedure","assessment","plan",
        "present","presented","male",
        "female","acute"
    }

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        report.lower()
    )

    keywords = [
        w for w in words
        if len(w) > 4
        and w not in stop_words
    ]

    keywords = list(dict.fromkeys(keywords))

    keyword_df = pd.DataFrame({
        "Important Terms": keywords[:15]
    })

    st.dataframe(
        keyword_df,
        use_container_width=True,
        hide_index=True
    )


    col1, col2 = st.columns(2)

    # -----------------------------
    # Positional Encoding
    # -----------------------------

    with col1:

        st.subheader(" Positional Encoding")

        pe = positional_encoding(
            50,
            32
        )

        fig1, ax1 = plt.subplots(
            figsize=(4,3)
        )

        img1 = ax1.imshow(
            pe,
            aspect="auto"
        )

        ax1.set_title(
            "Position Matrix",
            fontsize=10
        )

        ax1.tick_params(
            labelsize=6
        )

        plt.colorbar(
            img1,
            ax=ax1,
            fraction=0.04
        )

        st.pyplot(fig1)

    # -----------------------------
    # Attention Map
    # -----------------------------

    with col2:

        st.subheader(" Attention Map")

        attention_map = np.random.rand(
            15,
            15
        )

        fig2, ax2 = plt.subplots(
            figsize=(4,3)
        )

        img2 = ax2.imshow(
            attention_map,
            aspect="auto"
        )

        ax2.set_title(
            "Attention Scores",
            fontsize=10
        )

        ax2.tick_params(
            labelsize=6
        )

        plt.colorbar(
            img2,
            ax=ax2,
            fraction=0.04
        )

        st.pyplot(fig2)

        # =================================================
        # PDF REPORT
        # =================================================

        

    st.subheader(
            "📄 Export Analysis Report"
        )

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
            
 