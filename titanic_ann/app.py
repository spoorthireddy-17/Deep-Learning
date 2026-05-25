import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Titanic AI Prediction",
    page_icon="🚢",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

/* Background */

.stApp {
    background: linear-gradient(135deg, #0f172a, #111827, #1e293b);
    color: white;
}

/* Remove top padding */

.block-container {
    padding-top: 2rem;
}

/* Main Title */

.main-title {
    font-size: 50px;
    font-weight: 800;
    text-align: center;
    color: white;
    margin-bottom: 10px;
}

/* Subtitle */

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 22px;
    margin-bottom: 40px;
}

/* Glass Card */

.glass-card {
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 30px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.1);
}

/* Prediction Cards */

.success-card {
    background: linear-gradient(
        135deg,
        rgba(34,197,94,0.3),
        rgba(34,197,94,0.15)
    );

    padding: 25px;
    border-radius: 20px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: #22c55e;
}

.danger-card {
    background: linear-gradient(
        135deg,
        rgba(239,68,68,0.3),
        rgba(239,68,68,0.15)
    );

    padding: 25px;
    border-radius: 20px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: #ef4444;
}

/* Metric Cards */

.metric-card {
    background: rgba(255,255,255,0.06);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Button */

.stButton>button {

    width: 100%;
    height: 3.2em;

    border: none;
    border-radius: 12px;

    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    );

    color: white;
    font-size: 18px;
    font-weight: 600;

    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0 0 20px rgba(124,58,237,0.5);
}

/* Input Labels */

label {
    color: white !important;
    font-weight: 600 !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = tf.keras.models.load_model(
    "titanic_ann_model.keras"
)

# ---------------------------------------------------
# HERO SECTION
# ---------------------------------------------------

st.markdown("""
<div class="main-title">
🚢 Titanic Survival Prediction
</div>

<div class="subtitle">
Deep Learning Based Passenger Survival Prediction System
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TOP SECTION
# ---------------------------------------------------

left, right = st.columns([1.2,1])

# ---------------------------------------------------
# LEFT SIDE
# ---------------------------------------------------

with left:

    st.markdown("""
    <div class="glass-card">

    ## 📌 About The Project

    This AI-powered application predicts whether a passenger
    would survive during the Titanic disaster using a trained
    Artificial Neural Network (ANN).

    ### ⚡ Technologies Used
    - TensorFlow / Keras
    - Deep Learning
    - Streamlit
    - ANN Deployment

    ### 🎯 Features
    - Passenger Survival Prediction
    - Confidence Score
    - Interactive Dashboard

    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------
# RIGHT SIDE INPUT FORM
# ---------------------------------------------------

with right:

    st.markdown("""
    <div class="glass-card">
    """, unsafe_allow_html=True)

    st.subheader("🧾 Passenger Information")

    pclass = st.selectbox(
        "Passenger Class",
        [1,2,3]
    )

    age = st.slider(
        "Age",
        1,
        80,
        24
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=120.0
    )

    st.markdown("</div>", unsafe_allow_html=True)
    
    # ---------------------------------------------------
    # PREDICT BUTTON
    # ---------------------------------------------------

    predict = st.button("🚀 Predict Survival")

# ---------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------

pclass_norm = pclass / 5
age_norm = age / 100
fare_norm = fare / 150

input_data = np.array([
    [pclass_norm, age_norm, fare_norm]
])



# ---------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------

if predict:

    prediction = model.predict(input_data)

    probability = float(prediction[0][0])

    confidence = probability * 100

    st.write("")

    # ---------------------------------------------------
    # RESULT CARD
    # ---------------------------------------------------

    if probability > 0.5:

        st.markdown(f"""
        <div class="success-card">
        ✅ Passenger Likely To Survive
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="danger-card">
        ❌ Passenger Not Likely To Survive
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # ---------------------------------------------------
    # METRICS
    # ---------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(f"""
        <div class="metric-card">
        <h3>Survival Probability</h3>
        <h1>{probability:.2f}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="metric-card">
        <h3>Confidence Score</h3>
        <h1>{confidence:.2f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # ---------------------------------------------------
    # PROGRESS BAR
    # ---------------------------------------------------

    st.subheader("📈 Prediction Confidence")

    st.progress(int(confidence))

    # ---------------------------------------------------
    # SMALL MODERN CHART
    # ---------------------------------------------------

    survive_prob = probability
    nonsurvive_prob = 1 - probability

    st.write("")

    col1, col2, col3 = st.columns([1,2,1])

    with col2:

        fig, ax = plt.subplots(
            figsize=(3.5,3.5),
            facecolor='none'
        )

        ax.pie(
            [survive_prob, nonsurvive_prob],
            labels=["Survive", "Not Survive"],
            autopct='%1.1f%%',
            textprops={'color':'white'}
        )

        ax.set_facecolor('none')

        st.pyplot(
            fig,
            use_container_width=False
        )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.write("")

st.markdown("""
<center>
<p style='color:#94a3b8'>
Developed using TensorFlow + Streamlit
</p>
</center>
""", unsafe_allow_html=True)