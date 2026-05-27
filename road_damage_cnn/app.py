import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.preprocessing import image

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Road Damage Detection",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = tf.keras.models.load_model(
    "road_damage_model.h5"
)

# ---------------------------------------------------
# CLASS LABELS
# ---------------------------------------------------

class_names = [
    "Crack",
    "Manhole",
    "Pothole"
]

# ---------------------------------------------------
# SEVERITY FUNCTION
# ---------------------------------------------------

def get_severity(confidence):

    if confidence > 85:
        return "High"

    elif confidence > 60:
        return "Medium"

    else:
        return "Low"

# ---------------------------------------------------
# HEADER SECTION
# ---------------------------------------------------

st.title("AI-Based Road Damage Detection System")

st.subheader(
    "Smart City Infrastructure Monitoring using CNN"
)

st.markdown("---")

# ---------------------------------------------------
# ABOUT PROJECT
# ---------------------------------------------------

st.header("About the Project")

st.write("""
Road damage monitoring is essential for maintaining
safe transportation infrastructure in smart cities.

Manual road inspection is time-consuming and often
delays maintenance activities, increasing accident risks.

This project uses Convolutional Neural Networks (CNN)
to automatically classify road damages such as:
- potholes
- cracks
- manholes

Industry Applications:
- Smart City Monitoring
- Autonomous Vehicle Systems
- Municipal Road Maintenance
- Traffic Safety Systems
""")

st.markdown("---")

# ---------------------------------------------------
# IMAGE UPLOAD SECTION
# ---------------------------------------------------

st.header("Upload Road Image")

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# IMAGE PREVIEW
# ---------------------------------------------------

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.header("Uploaded Image Preview")

    st.image(
        img,
        caption="Uploaded Road Image",
        use_container_width=True
    )

    # ---------------------------------------------------
    # PREPROCESS IMAGE
    # ---------------------------------------------------

    img_resized = img.resize((224,224))

    img_array = image.img_to_array(
        img_resized
    )

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # ---------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = np.max(prediction) * 100

    severity = get_severity(confidence)

    # ---------------------------------------------------
    # PREDICTION AREA
    # ---------------------------------------------------

    st.header("Prediction Result")

    st.success(
        f"Prediction: {predicted_class} Detected"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    st.warning(
        f"Severity Level: {severity}"
    )

    st.markdown("---")

    # ---------------------------------------------------
    # VISUALIZATION AREA
    # ---------------------------------------------------

    st.header("Class Confidence Visualization")

    probs = prediction[0] * 100

    fig, ax = plt.subplots()

    ax.bar(class_names, probs)

    ax.set_ylabel("Confidence %")

    ax.set_xlabel("Damage Type")

    ax.set_title("Prediction Confidence")

    st.pyplot(fig)

    st.markdown("---")

    # ---------------------------------------------------
    # RECOMMENDATIONS
    # ---------------------------------------------------

    st.header("Recommendations")

    if predicted_class == "Pothole":

        st.error("""
Immediate maintenance recommended.

High-risk road condition detected.
Potential vehicle damage and accident risk.
""")

    elif predicted_class == "Crack":

        st.warning("""
Preventive maintenance suggested.

Road surface deterioration detected.
Repair recommended before worsening.
""")

    elif predicted_class == "Manhole":

        st.info("""
Inspect manhole alignment and surface condition.

Routine infrastructure maintenance recommended.
""")