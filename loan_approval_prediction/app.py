import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide"
)

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.image(
    "https://img.icons8.com/color/96/bank-building.png",
    width=90
)

st.sidebar.title("🏦 Loan Approval System")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Loan Prediction",
        "📊 About Model",
        "ℹ️ Project Info"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    AI-powered loan approval prediction
    using PCA, K-Means and Random Forest.
    """
)

# ==========================================
# LOAD MODELS
# ==========================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

rf = joblib.load(os.path.join(BASE_DIR, "random_forest_small.pkl"))
pca = joblib.load(os.path.join(BASE_DIR, "pca.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
kmeans = joblib.load(os.path.join(BASE_DIR, "kmeans.pkl"))

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(
        135deg,
        #f8fafc 0%,
        #e2e8f0 50%,
        #cbd5e1 100%
    );
}

/* Main Title */
.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 10px;
}

/* Subtitle */
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #475569;
    margin-bottom: 30px;
}

/* Form Container */
div[data-testid="stForm"] {
    background: white;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.1);
}

/* Labels */
label {
    color: #0f172a !important;
    font-weight: 600 !important;
}

/* Input boxes */
.stNumberInput input,
.stSelectbox div,
.stTextInput input {
    color: #0f172a !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: white;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}

/* Result Cards */
.result-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
}

/* Button */
.stButton > button,
.stFormSubmitButton > button {
    width: 100%;
    background: #16a34a;
    color: white;
    font-size: 18px;
    font-weight: 600;
    border-radius: 10px;
    border: none;
    padding: 12px;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #15803d;
}

/* Headers */
h1, h2, h3 {
    color: #0f172a !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="main-title">
🏦 Loan Approval Prediction System
</div>

<div class="sub-title">
AI-Powered Credit Risk Assessment & Loan Eligibility Analysis
</div>
""", unsafe_allow_html=True)


#home page
if page == "🏠 Home":

    st.write("""
    Welcome to the Loan Approval Prediction Dashboard.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🏦 Model", "Random Forest")

    with col2:
        st.metric("📊 Features", "17")

    with col3:
        st.metric("⚡ Status", "Ready")

    st.markdown("---")

    st.success(
        "Predict loan approval and assess applicant risk using Machine Learning."
    )
# ==========================================
# FORM
# ==========================================

if page == "🔍 Loan Prediction":

    with st.form("loan_form"):

        st.subheader("👤 Applicant Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "🎂 Age",
                min_value=18,
                max_value=100,
                value=30
            )

            income = st.number_input(
                "💰 Annual Income ($)",
                min_value=0.0,
                value=50000.0
            )

            credit_score = st.number_input(
                "📊 Credit Score",
                min_value=300,
                max_value=850,
                value=700,
                help="Higher score indicates better creditworthiness"
            )

        with col2:

            months_employed = st.number_input(
                "💼 Months Employed",
                min_value=0,
                value=24
            )

            num_credit_lines = st.number_input(
                "💳 Number of Credit Lines",
                min_value=0,
                value=5
            )

            employment = st.selectbox(
                "🏢 Employment Type",
                ["Full-time", "Part-time", "Self-employed", "Unemployed"]
            )

            education = st.selectbox(
                "🎓 Education",
                ["High School", "Bachelor", "Master", "PhD"]
            )

        with col3:

            marital = st.selectbox(
                "💍 Marital Status",
                ["Single", "Married", "Divorced"]
            )

            dependents = st.selectbox(
                "👨‍👩‍👧 Has Dependents",
                [0, 1]
            )

            mortgage = st.selectbox(
                "🏠 Has Mortgage",
                [0, 1]
            )

        st.markdown("---")

        st.subheader("🏦 Loan Information")

        col4, col5, col6 = st.columns(3)

        with col4:

            loan_amount = st.number_input(
                "💵 Loan Amount ($)",
                min_value=0.0,
                value=10000.0
            )

            loan_term = st.number_input(
                "📅 Loan Term (Months)",
                min_value=1,
                value=36
            )

        with col5:

            interest_rate = st.number_input(
                "📈 Interest Rate (%)",
                min_value=0.0,
                value=8.5
            )

            dti_ratio = st.slider(
                "⚖️ Debt-To-Income Ratio",
                0.0,
                1.0,
                0.30
            )

        with col6:

            purpose = st.selectbox(
                "🎯 Loan Purpose",
                ["Home", "Auto", "Education", "Business", "Other"]
            )

            cosigner = st.selectbox(
                "🤝 Has Co-Signer",
                [0, 1]
            )

        st.markdown("<br>", unsafe_allow_html=True)

        submit = st.form_submit_button(
            "🚀 Predict Loan Approval",
            use_container_width=True
        )

    # ==========================================
    # PREDICTION
    # ==========================================

    if submit:

        # IMPORTANT:
        # Must match training encodings

        education_map = {
            "High School": 0,
            "Bachelor": 1,
            "Master": 2,
            "PhD": 3
        }

        employment_map = {
            "Full-time": 0,
            "Part-time": 1,
            "Self-employed": 2,
            "Unemployed": 3
        }

        marital_map = {
            "Single": 0,
            "Married": 1,
            "Divorced": 2
        }

        purpose_map = {
            "Home": 0,
            "Auto": 1,
            "Education": 2,
            "Business": 3,
            "Other": 4
        }

        input_df = pd.DataFrame([[
            0,                      # LoanID
            age,
            income,
            loan_amount,
            credit_score,
            months_employed,
            num_credit_lines,
            interest_rate,
            loan_term,
            dti_ratio,
            education_map[education],
            employment_map[employment],
            marital_map[marital],
            mortgage,
            dependents,
            purpose_map[purpose],
            cosigner
        ]], columns=[
            'LoanID',
            'Age',
            'Income',
            'LoanAmount',
            'CreditScore',
            'MonthsEmployed',
            'NumCreditLines',
            'InterestRate',
            'LoanTerm',
            'DTIRatio',
            'Education',
            'EmploymentType',
            'MaritalStatus',
            'HasMortgage',
            'HasDependents',
            'LoanPurpose',
            'HasCoSigner'
        ])

        # Scale
        scaled = scaler.transform(input_df)

        # PCA
        transformed = pca.transform(scaled)

        # KMeans Cluster
        segment = kmeans.predict(transformed)

        # Final RF input
        final_input = pd.DataFrame(
            transformed,
            columns=[str(i) for i in range(transformed.shape[1])]
        )

        final_input["Segment"] = segment

        # Prediction
            # Prediction
        prediction = rf.predict(final_input)[0]
        probability = rf.predict_proba(final_input)[0][1]

        st.markdown("---")

        st.markdown("## 📊 Loan Assessment Report")

        # Top Metrics
        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric(
                "⚠️ Default Risk",
                f"{probability:.2%}"
            )

        with m2:
            st.metric(
                "👥 Customer Segment",
                int(segment[0])
            )

        with m3:
            approval_confidence = (1 - probability) * 100

            st.metric(
                "🎯 Approval Confidence",
                f"{approval_confidence:.2f}%"
            )

        st.markdown("---")

    # Result Card

        if prediction == 0:

            st.success("✅ LOAN APPROVED")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Approval Confidence",
                    f"{(1-probability)*100:.2f}%"
                )

            with col2:
                st.metric(
                    "Default Risk",
                    f"{probability*100:.2f}%"
                )

            st.info(
                "Applicant demonstrates strong creditworthiness and low default risk."
            )

        else:

            st.error("❌ LOAN REJECTED")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Risk Score",
                    f"{probability*100:.2f}%"
                )

            with col2:
                st.metric(
                    "Approval Confidence",
                    f"{(1-probability)*100:.2f}%"
                )

            st.warning(
                "Applicant has elevated default probability. Further review is recommended."
            )

        st.markdown("---")

        # Risk Analysis

        st.subheader("📈 Risk Analysis")

        st.progress(float(probability))

        if probability < 0.30:

            st.success(
                "🟢 Low Risk Applicant - Highly Eligible"
            )

            recommendation = "Strongly Recommended"

        elif probability < 0.60:

            st.warning(
                "🟡 Moderate Risk Applicant - Review Required"
            )

            recommendation = "Needs Additional Review"

        else:

            st.error(
                "🔴 High Risk Applicant - Not Recommended"
            )

            recommendation = "High Risk"

        st.subheader("📝 Lending Recommendation")

        st.info(recommendation)

        # Applicant Summary

        st.subheader("📋 Applicant Summary")

        summary = pd.DataFrame({
            "Feature": [
                "Age",
                "Income",
                "Loan Amount",
                "Credit Score",
                "Months Employed",
                "Credit Lines",
                "Interest Rate",
                "Loan Term"
            ],
            "Value": [
                age,
                income,
                loan_amount,
                credit_score,
                months_employed,
                num_credit_lines,
                interest_rate,
                loan_term
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )
        
if page == "📊 About Model":

    st.header("📊 Machine Learning Pipeline")

    st.markdown("""
    ### Algorithms Used

    ✅ PCA – Feature Reduction

    ✅ K-Means – Customer Segmentation

    ✅ Random Forest – Risk Prediction

    ✅ Logistic Regression – Classification
    """)

    st.info(
        "The system predicts whether an applicant is likely to default."
    )
    
if page == "ℹ️ Project Info":

    st.header("ℹ️ Project Information")

    st.markdown("""
    ### Dataset
    Loan Default Prediction Dataset

    ### Objective
    Predict whether a loan should be approved based on applicant information.

    ### Features
    - Age
    - Income
    - Credit Score
    - Loan Amount
    - Employment History
    - Interest Rate
    - DTI Ratio
    """)

    st.info(
        "Built using Streamlit, Scikit-Learn, PCA, K-Means and Random Forest."
    )
    
st.markdown("---")

st.markdown(
    """
    <div style='text-align:center; color:gray;'>
    🏦 Loan Approval Prediction System 
    </div>
    """,
    unsafe_allow_html=True
)