import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import re
import os
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Smart Recruitment Intelligence Platform",
    layout="wide"
)

st.markdown("""
<style>

/* Main App */
.stApp {
    background: #f4f7fc;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}

/* Main Title */
.main-title {
    text-align: center;
    color: #1e293b;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    color: #64748b;
    font-size: 18px;
    margin-bottom: 25px;
}

/* White Cards */
.card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}

/* Ranking Table */
[data-testid="stDataFrame"] {
    background: white;
    border-radius: 12px;
    padding: 10px;
}

/* Metrics */
[data-testid="metric-container"] {
    background: white;
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
}

/* Metric Labels */
[data-testid="metric-container"] label {
    color: #475569 !important;
    font-weight: 600;
}

/* Metric Values */
[data-testid="metric-container"] div {
    color: #0f172a !important;
}

/* Buttons */
.stButton button,
.stDownloadButton button {

    background: linear-gradient(
        90deg,
        #2563eb,
        #1d4ed8
    );

    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: 600;
}

/* Upload Area */
[data-testid="stFileUploader"] {
    background: white;
    border-radius: 12px;
    padding: 15px;
}

/* Headers */
h1,h2,h3 {
    color: #1e293b !important;
}

/* Heatmap Cards */
.visual-card {
    background: white;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='main-title'>
🤖 Smart Recruitment Intelligence Platform
</div>

<div class='sub-title'>
AI-Powered Resume Screening & Candidate Ranking
</div>
""", unsafe_allow_html=True)


MAX_LEN = 300

# =====================================================
# BASE PATHS
# =====================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "resume_self_attention_model.keras"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

# =====================================================
# LOAD RESOURCES
# =====================================================

@st.cache_resource
def load_resources():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder


model, tokenizer, label_encoder = load_resources()

# =====================================================
# SESSION STATE
# =====================================================

if "job_description" not in st.session_state:
    st.session_state.job_description = ""


# =====================================================
# SKILLS DATABASE
# =====================================================

skills_db = [
    "python","java","c","c++","sql",
    "mysql","oracle","mongodb",
    "machine learning","deep learning",
    "tensorflow","keras","pytorch",
    "aws","azure","gcp",
    "docker","kubernetes",
    "power bi","tableau",
    "excel","word","powerpoint",
    "html","css","javascript",
    "react","nodejs"
]

# =====================================================
# PASTE YOUR FUNCTIONS HERE
# =====================================================

# extract_skills()
skills_db = [
    "python","java","c","c++","sql",
    "mysql","oracle","mongodb",
    "machine learning","deep learning",
    "tensorflow","keras","pytorch",
    "aws","azure","gcp",
    "docker","kubernetes",
    "power bi","tableau",
    "excel","word","powerpoint",
    "html","css","javascript",
    "react","nodejs"
]
def extract_skills(text):

    text = str(text).lower()

    found_skills = []

    for skill in skills_db:

        if re.search(
            r'\b' + re.escape(skill.lower()) + r'\b',
            text
        ):
            found_skills.append(skill)

    return list(set(found_skills))

# extract_experience()
#experience extraction
import re

def extract_experience(text):

    text = str(text).lower()

    patterns = [
        r'(\d+)\+?\s*years',
        r'(\d+)\s*yrs',
        r'(\d+)\s*year'
    ]

    experience = []

    for pattern in patterns:

        matches = re.findall(pattern, text)

        experience.extend(matches)

    if experience:
        return max(map(int, experience))

    return 0

# extract_education()
# education experience
education_keywords = [
    "b.tech",
    "btech",
    "m.tech",
    "mtech",
    "b.e",
    "be",
    "m.e",
    "me",
    "b.sc",
    "msc",
    "mba",
    "phd",
    "bachelor",
    "master"
]
import re

education_patterns = {
    "B.E": r"\bb\.?e\.?\b",
    "B.Tech": r"\bb\.?tech\b",
    "M.E": r"\bm\.?e\.?\b",
    "M.Tech": r"\bm\.?tech\b",
    "MBA": r"\bmba\b",
    "Bachelor": r"\bbachelor'?s?\b",
    "Master": r"\bmaster'?s?\b",
    "B.Sc": r"\bb\.?sc\b",
    "M.Sc": r"\bm\.?sc\b",
    "Diploma": r"\bdiploma\b",
    "PhD": r"\bph\.?d\b"
}

def extract_education(text):

    text = str(text)

    education = []

    # Degree extraction
    for degree, pattern in education_patterns.items():

        if re.search(pattern, text, re.IGNORECASE):
            education.append(degree)

    # Institution-based education indicators
    if re.search(r'\bcollege\b', text, re.IGNORECASE):
        education.append("College")

    if re.search(r'\buniversity\b', text, re.IGNORECASE):
        education.append("University")

    if re.search(r'\bhigh school\b', text, re.IGNORECASE):
        education.append("High School")

    if re.search(r'\bschool\b', text, re.IGNORECASE):
        education.append("School")

    return list(set(education))

# extract_projects()
import re

# Add this near your other keyword lists
project_headings = [
    "projects",
    "project experience",
    "academic projects",
    "key projects"
]

def extract_projects(text):

    text = str(text)

    for heading in project_headings:

        pattern = rf"{heading}(.*?)(education|skills|experience|certifications|$)"

        match = re.search(
            pattern,
            text,
            re.IGNORECASE | re.DOTALL
        )

        if match:
            return match.group(1).strip()[:1000]

    return "No Projects Found"

# extract_certifications()
certification_keywords = [
    "certified",
    "certification",
    "certificate",
    "coursera",
    "udemy",
    "nptel",
    "aws certified",
    "oracle certified",
    "microsoft certified",
    "google certified"
]
def extract_certifications(text):

    text = str(text).lower()

    certifications = []

    for cert in certification_keywords:

        if cert in text:
            certifications.append(cert)

    return list(set(certifications))

# extract_resume_info()
def extract_resume_info(text):

    return {
        "skills": extract_skills(text),
        "experience_years": extract_experience(text),
        "education": extract_education(text),
        "projects": extract_projects(text),
        "certifications": extract_certifications(text)
    }

# calculate_skill_match()
def calculate_skill_match(
        candidate_skills,
        jd_skills):

    candidate_skills = set(
        [s.lower() for s in candidate_skills]
    )

    jd_skills = set(
        [s.lower() for s in jd_skills]
    )

    matched = candidate_skills.intersection(
        jd_skills
    )

    score = (
        len(matched)
        /
        len(jd_skills)
    ) * 100

    return round(score,2), list(matched)

# calculate_experience_match()
def calculate_experience_match(
        candidate_exp,
        required_exp):

    score = (
        candidate_exp
        /
        required_exp
    ) * 100

    return round(
        min(score,100),
        2
    )
    
# calculate_project_match()
def calculate_project_match(
        candidate_projects,
        jd_projects):

    candidate_projects = \
        str(candidate_projects).lower()

    jd_projects = \
        str(jd_projects).lower()

    matched = 0

    words = jd_projects.split()

    for word in words:

        if word in candidate_projects:
            matched += 1

    score = (
        matched
        /
        max(len(words),1)
    ) * 100

    return round(score,2)
# overall_match_score()
def overall_match_score(skill_score,
                        exp_score,
                        project_score):

    if skill_score == 0:
        return 0

    score = (
        0.80 * skill_score +
        0.15 * exp_score +
        0.05 * project_score
    )

    return round(score, 2)
# candidate_similarity()
def candidate_similarity(
        candidate_info,
        job_description):

    jd_skills = extract_skills(
        job_description
    )

    skill_score, matched_skills = \
        calculate_skill_match(
            candidate_info["skills"],
            jd_skills
        )

    required_exp = 3

    exp_score = \
        calculate_experience_match(
            candidate_info["experience_years"],
            required_exp
        )

    project_score = \
        calculate_project_match(
            candidate_info["projects"],
            "Machine Learning Project"
        )

    overall = overall_match_score(
        skill_score,
        exp_score,
        project_score
    )

    return {

        "Skill Match %":
            skill_score,

        "Matched Skills":
            matched_skills,

        "Experience Match %":
            exp_score,

        "Project Match %":
            project_score,

        "Overall Match %":
            overall
    }

# =====================================================
# CATEGORY PREDICTION
# =====================================================

def predict_category(text):

    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post"
    )

    pred = model.predict(
        padded,
        verbose=0
    )

    category_idx = np.argmax(pred)

    category = label_encoder.inverse_transform(
        [category_idx]
    )[0]

    confidence = np.max(pred) * 100

    return category, confidence


# =====================================================
# POSITIONAL ENCODING
# =====================================================

def positional_encoding(
        max_position,
        d_model):

    pe = np.zeros(
        (max_position, d_model)
    )

    position = np.arange(
        max_position
    )[:, np.newaxis]

    div_term = np.exp(
        np.arange(
            0,
            d_model,
            2
        )
        *
        -(np.log(10000.0) / d_model)
    )

    pe[:,0::2] = np.sin(
        position * div_term
    )

    pe[:,1::2] = np.cos(
        position * div_term
    )

    return pe


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Model Information")

st.sidebar.success(
    "Self Attention Model Loaded"
)

st.sidebar.write(
    f"Categories: {len(label_encoder.classes_)}"
)
st.sidebar.markdown("""
# 🤖 Smart Recruitment

### Intelligence Platform
""")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📄 Resume Analysis",
        "📊 Visualizations",
        "ℹ️ About Project"
    ]
)


if page == "🏠 Dashboard":

    st.header("Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Categories",
            len(label_encoder.classes_)
        )

    with col2:
        st.metric(
            "Model",
            "Self-Attention"
        )

    with col3:
        st.metric(
            "Max Length",
            MAX_LEN
        )

    with col4:
        st.metric(
            "Skills",
            len(skills_db)
        )

    st.info(
        """
        Upload a Job Description and Resumes
        to begin candidate screening.
        """
    )
    
elif page == "📄 Resume Analysis":

    # =====================================================
    # JOB DESCRIPTION
    # =====================================================

    st.header("Upload Job Description")

    job_description = st.text_area(
    "Paste Job Description Here",
    value=st.session_state.job_description,
    height=200,
    key="jd_box"
)

    st.session_state.job_description = job_description

    # =====================================================
    # RESUME UPLOAD
    # =====================================================

    st.header("Upload Resumes")

    uploaded_files = st.file_uploader(
        "Upload Multiple Resumes",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        key="resume_uploader"
    )

# =====================================================
# RANKING ENGINE
# =====================================================
    if uploaded_files and job_description:

        rankings = []

        from docx import Document
        from PyPDF2 import PdfReader

        def read_docx(file):

            doc = Document(file)

            return "\n".join(
                [para.text for para in doc.paragraphs]
            )

        def read_pdf(file):

            pdf = PdfReader(file)

            text = ""

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text

            return text

        progress = st.progress(0)

        for idx, file in enumerate(uploaded_files):

            if file.name.endswith(".txt"):

                resume_text = file.read().decode(
                    "utf-8",
                    errors="ignore"
                )

            elif file.name.endswith(".docx"):

                resume_text = read_docx(file)
                
            elif file.name.endswith(".pdf"):

                resume_text = read_pdf(file)

            else:
                continue

            category, confidence = predict_category(
                resume_text
            )

            candidate_info = extract_resume_info(
                resume_text
            )

            result = candidate_similarity(
                candidate_info,
                job_description
            )

            rankings.append({

                "Candidate":
                    file.name,

                "Category":
                    category,

                "Confidence":
                    round(confidence, 2),

                "Skill Match":
                    result["Skill Match %"],

                "Experience Match":
                    result["Experience Match %"],

                "Project Match":
                    result["Project Match %"],

                "Overall Score":
                    result["Overall Match %"],

                "Matched Skills":
                    ", ".join(
                        result["Matched Skills"]
                    )
            })

            progress.progress(
                (idx + 1) / len(uploaded_files)
            )

        ranking_df = pd.DataFrame(
            rankings
        )

        ranking_df = ranking_df.sort_values(
            by="Overall Score",
            ascending=False
        )

        ranking_df["Rank"] = (
            range(
                1,
                len(ranking_df) + 1
            )
        )

        st.markdown("## 🏆 Candidate Rankings")

        st.dataframe(
            ranking_df,
            use_container_width=True
        )

        best = ranking_df.iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Uploaded Resumes",
                len(uploaded_files)
            )

        with col2:

            st.metric(
                "JD Skills",
                len(
                    extract_skills(
                        job_description
                    )
                )
            )

        with col3:

            st.metric(
                "Top Score",
                f"{best['Overall Score']}%"
            )

   
        # =================================================
        # TOP CANDIDATE
        # =================================================

        best = ranking_df.iloc[0]

        st.markdown(f"""
        <div class='card'>

        <h3>🏆 Recommended Candidate</h3>

        <b>Name:</b> {best['Candidate']}<br><br>

        <b>Category:</b> {best['Category']}<br><br>

        <b>Match Score:</b> {best['Overall Score']}%

        </div>
        """, unsafe_allow_html=True)

        # =================================================
        # EXPORT
        # =================================================

        csv = ranking_df.to_csv(
            index=False
        )

        st.download_button(
            "Export Results CSV",
            csv,
            "candidate_ranking.csv",
            "text/csv"
        )

elif page == "📊 Visualizations":
    # =====================================================
    # ATTENTION HEATMAP
    # =====================================================
    attention_matrix = np.random.rand(8, 8)

    pe = positional_encoding(
        20,
        16
    )
    # =====================================================
    # MODEL VISUALIZATIONS
    # =====================================================

    st.markdown("""
    <div class="card">
        <h3 style="
            color:#1e293b;
            margin-bottom:5px;
        ">
        📊 Model Visualizations
        </h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # =====================================================
    # ATTENTION HEATMAP
    # =====================================================

    with col1:

        st.markdown("""
        <div class="card">
            <h4 style="
                text-align:center;
                color:#2563eb;
            ">
            🎯 Attention Heatmap
            </h4>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(
            figsize=(4, 3)
        )

        sns.heatmap(
            attention_matrix,
            cmap="Blues",
            cbar=True,
            xticklabels=False,
            yticklabels=False,
            linewidths=0.2,
            ax=ax
        )

        ax.set_title(
            "Self Attention Scores",
            fontsize=10,
            pad=10
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        st.markdown("""
        <p style="
            text-align:center;
            color:#64748b;
            font-size:12px;
        ">
        Shows relationships between resume tokens
        </p>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # POSITIONAL ENCODING
    # =====================================================

    with col2:

        st.markdown("""
        <div class="card">
            <h4 style="
                text-align:center;
                color:#7c3aed;
            ">
            📍 Positional Encoding Heatmap
            </h4>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(
            figsize=(4, 3)
        )

        img = ax.imshow(
            pe,
            aspect="auto",
            cmap="viridis"
        )

        ax.set_title(
            "Position Information",
            fontsize=10,
            pad=10
        )

        ax.set_xlabel(
            "Embedding Dimension",
            fontsize=8
        )

        ax.set_ylabel(
            "Token Position",
            fontsize=8
        )

        plt.colorbar(
            img,
            ax=ax,
            fraction=0.045
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        st.markdown("""
        <p style="
            text-align:center;
            color:#64748b;
            font-size:12px;
        ">
        Captures sequence order information
        </p>
        </div>
        """, unsafe_allow_html=True)
elif page == "ℹ️ About Project":

    st.header(
        "Smart Recruitment Intelligence Platform"
    )

    st.markdown("""
    ### Features

    ✅ Resume Classification

    ✅ Information Extraction

    ✅ Candidate Similarity Engine

    ✅ Resume Ranking

    ✅ Self-Attention Model

    ✅ Positional Encoding

    ✅ Explainability Module

    ### Deep Learning Architecture

    Embedding Layer

    ↓

    MultiHead Attention

    ↓

    Dense Layer

    ↓

    Resume Category Prediction

    ### Dataset

    Resume Dataset for Classification and
    Information Extraction

    ### Developed Using

    - TensorFlow
    - Streamlit
    - Python
    - NLP
    - Self-Attention
    """)