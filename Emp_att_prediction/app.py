import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.markdown("""
<style>

/* Main Background Gradient */

.stApp {
    background: linear-gradient(
        135deg,
        #F8FAFC 0%,
        #EEF4FF 40%,
        #E0EAFC 100%
    );
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #FFFFFF 0%,
        #F1F5F9 100%
    );
    border-right: 1px solid #E2E8F0;
}

/* Main Title */

.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #0F172A;
}

/* Subtitle */

.subtitle {
    font-size: 18px;
    color: #64748B;
}

/* Metric Cards */

.metric-card {
    background: linear-gradient(
        135deg,
        #FFFFFF 0%,
        #F8FAFC 100%
    );

    padding: 25px;
    border-radius: 18px;

    box-shadow:
        0px 6px 20px rgba(0,0,0,0.08);

    text-align: center;
}

/* Prediction Card */

.prediction-card {

    background: linear-gradient(
        135deg,
        #FFFFFF 0%,
        #F8FAFC 100%
    );

    border-radius: 18px;

    padding: 25px;

    box-shadow:
        0px 8px 25px rgba(0,0,0,0.08);
}

/* Button */

.stButton > button {

    background: linear-gradient(
        90deg,
        #2563EB,
        #4F46E5
    );

    color: white;

    border: none;

    border-radius: 12px;

    font-weight: 600;

    padding: 12px 30px;

    width: 100%;
}

.stButton > button:hover {

    background: linear-gradient(
        90deg,
        #1D4ED8,
        #4338CA
    );

    color: white;
}

/* Risk Cards */

.high-risk-card {

    background: linear-gradient(
        135deg,
        #FEE2E2,
        #FCA5A5
    );

    padding: 20px;

    border-radius: 15px;

    text-align: center;
}

.medium-risk-card {

    background: linear-gradient(
        135deg,
        #FFEDD5,
        #FDBA74
    );

    padding: 20px;

    border-radius: 15px;

    text-align: center;
}

.low-risk-card {

    background: linear-gradient(
        135deg,
        #DCFCE7,
        #86EFAC
    );

    padding: 20px;

    border-radius: 15px;

    text-align: center;
}

</style>
""", unsafe_allow_html=True)
# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# LOAD DATA
# =====================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )

df = load_data()

# =====================================
# LOAD MODEL
# =====================================

model = joblib.load(
    "random_forest_streamlit.pkl"
)

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Home",
        "🤖 Attrition Prediction",
        "📈 EDA Dashboard",
        "👥 Employee Segmentation"
    ]
)

# =====================================
# HOME PAGE
# =====================================

if page == "🏠 Home":

    st.title(
        "AI-Powered Employee Attrition Prediction System"
    )

    st.markdown("""
    ### Project Objective

    Predict employees likely to leave the organization
    and help HR:

    - Identify high-risk employees
    - Improve employee retention
    - Reduce turnover costs
    - Support HR decision making
    """)

    col1, col2, col3 = st.columns(3)

    attrition_rate = (
        df["Attrition"]
        .value_counts(normalize=True)["Yes"]
        * 100
    )

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{len(df)}</h2>
            <p>Total Employees</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{df.shape[1]}</h2>
            <p>Features</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{attrition_rate:.2f}%</h2>
            <p>Attrition Rate</p>
        </div>
        """, unsafe_allow_html=True)

   

# =====================================
# PREDICTION PAGE
# =====================================

elif page == "🤖 Attrition Prediction":

    st.markdown("""
    <div class='main-title'>
    Employee Attrition Risk Assessment
    </div>

    <div class='subtitle'>
    Enter employee information to predict attrition probability and risk level.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 👤 Employee Information")

        age = st.slider(
            "Age",
            18,
            60,
            30
        )

        monthly_income = st.number_input(
            "Monthly Income ($)",
            min_value=1000,
            value=5000
        )

        job_satisfaction = st.selectbox(
            "Job Satisfaction",
            [1, 2, 3, 4],
            help="1 = Low, 4 = High"
        )

        environment_satisfaction = st.selectbox(
            "Environment Satisfaction",
            [1, 2, 3, 4],
            help="1 = Low, 4 = High"
        )

        overtime = st.selectbox(
            "OverTime",
            ["No", "Yes"]
        )

    with col2:

        st.markdown("### 💼 Work Information")

        work_life_balance = st.selectbox(
            "Work Life Balance",
            [1, 2, 3, 4],
            help="1 = Poor, 4 = Excellent"
        )

        years_at_company = st.slider(
            "Years At Company",
            0,
            40,
            5
        )

        total_working_years = st.slider(
            "Total Working Years",
            0,
            40,
            8
        )

        job_level = st.slider(
            "Job Level",
            1,
            5,
            2
        )

        distance_from_home = st.slider(
            "Distance From Home (KM)",
            1,
            30,
            10
        )

    st.markdown("")

    predict = st.button(
        "🔍 Predict Attrition Risk",
        use_container_width=True
    )

    if predict:

        overtime_encoded = (
            1 if overtime == "Yes" else 0
        )

        input_data = np.array([[
            age,
            monthly_income,
            job_satisfaction,
            environment_satisfaction,
            overtime_encoded,
            work_life_balance,
            years_at_company,
            total_working_years,
            job_level,
            distance_from_home
        ]])

        probability = (
            model.predict_proba(input_data)[0][1]
        ) * 100

        if probability >= 70:
            risk = "HIGH"

        elif probability >= 40:
            risk = "MEDIUM"

        else:
            risk = "LOW"

        st.markdown("---")

        st.subheader("📊 Prediction Results")

        kpi1, kpi2 = st.columns(2)

        with kpi1:

            st.metric(
                label="Attrition Probability",
                value=f"{probability:.2f}%"
            )

        with kpi2:

            if risk == "LOW":

                st.markdown("""
                <div style="
                    background:#DCFCE7;
                    color:#15803D;
                    padding:20px;
                    border-radius:12px;
                    text-align:center;
                    font-size:24px;
                    font-weight:bold;">
                    🟢 LOW RISK
                </div>
                """,
                unsafe_allow_html=True)

            elif risk == "MEDIUM":

                st.markdown("""
                <div style="
                    background:#FFEDD5;
                    color:#C2410C;
                    padding:20px;
                    border-radius:12px;
                    text-align:center;
                    font-size:24px;
                    font-weight:bold;">
                    🟠 MEDIUM RISK
                </div>
                """,
                unsafe_allow_html=True)

            else:

                st.markdown("""
                <div style="
                    background:#FEE2E2;
                    color:#B91C1C;
                    padding:20px;
                    border-radius:12px;
                    text-align:center;
                    font-size:24px;
                    font-weight:bold;">
                    🔴 HIGH RISK
                </div>
                """,
                unsafe_allow_html=True)

        st.markdown("")

        st.subheader("📋 HR Recommendations")

        if risk == "HIGH":

            st.error("""
            • Schedule immediate retention discussion

            • Review employee compensation package

            • Reduce overtime workload

            • Provide career growth opportunities

            • Conduct employee satisfaction survey
            """)

        elif risk == "MEDIUM":

            st.warning("""
            • Monitor employee engagement regularly

            • Conduct manager feedback sessions

            • Offer skill development programs

            • Review work-life balance concerns
            """)

        else:

            st.success("""
            • Employee retention appears healthy

            • Continue recognition programs

            • Maintain engagement initiatives

            • Encourage professional growth
            """)

        st.subheader("🎯 Risk Summary")

        st.info(
            f"""
            Employee Attrition Probability: {probability:.2f}%

            Risk Classification: {risk}

            AI Recommendation Generated Successfully.
            """
        )


# =====================================
# EDA DASHBOARD
# =====================================

elif page == "📈 EDA Dashboard":

    st.title(
        "📈 Employee Attrition Dashboard"
    )

    attrition_count = (
        df["Attrition"]
        .value_counts()
        .reset_index()
    )

    attrition_count.columns = [
        "Attrition",
        "Count"
    ]

    fig1 = px.pie(
        attrition_count,
        values="Count",
        names="Attrition",
        title="Attrition Distribution"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    fig2 = px.box(
        df,
        x="Attrition",
        y="MonthlyIncome",
        title="Salary Impact on Attrition"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    fig3 = px.histogram(
        df,
        x="Age",
        color="Attrition",
        title="Age Distribution"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    dept_df = (
        df.groupby("Department")
        .size()
        .reset_index(name="Employees")
    )

    fig4 = px.bar(
        dept_df,
        x="Department",
        y="Employees",
        title="Department Distribution"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

# =====================================
# SEGMENTATION PAGE
# =====================================

# =====================================
# EMPLOYEE SEGMENTATION PAGE
# =====================================

elif page == "👥 Employee Segmentation":

    st.markdown("""
    <div class='main-title'>
    Employee Segmentation Analysis
    </div>

    <div class='subtitle'>
    K-Means clustering groups employees into meaningful categories for HR decision making.
    </div>
    """, unsafe_allow_html=True)

    # Load cluster data

    cluster_data = pd.read_csv(
        "employee_clusters.csv"
    )

    st.markdown("---")

    # Cluster Counts

    cluster_counts = (
        cluster_data["Cluster"]
        .value_counts()
        .reset_index()
    )

    cluster_counts.columns = [
        "Cluster",
        "Employees"
    ]

    # KPI Cards

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{cluster_counts.iloc[0]['Employees']}</h2>
            <p>Cluster 0</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{cluster_counts.iloc[1]['Employees']}</h2>
            <p>Cluster 1</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{cluster_counts.iloc[2]['Employees']}</h2>
            <p>Cluster 2</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 Cluster Distribution")

    fig = px.pie(
        cluster_counts,
        names="Cluster",
        values="Employees",
        hole=0.4,
        title="Employee Segmentation"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("### 📋 Cluster Interpretation")

    cluster_info = pd.DataFrame({
        "Cluster": [
            "Cluster 0",
            "Cluster 1",
            "Cluster 2"
        ],
        "Employee Group": [
            "High Performers",
            "At Risk Employees",
            "New Employees"
        ],
        "Characteristics": [
            "High Salary, High Satisfaction",
            "Low Satisfaction, High Attrition Risk",
            "Low Experience, High Turnover Risk"
        ]
    })

    st.dataframe(
        cluster_info,
        use_container_width=True
    )

    st.markdown("### 🎯 HR Recommendations")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("""
        🟢 High Performers

        • Reward top performers

        • Leadership programs

        • Career advancement
        """)

    with col2:
        st.warning("""
        🟠 At Risk Employees

        • Retention discussions

        • Salary review

        • Work-life balance support
        """)

    with col3:
        st.info("""
        🔵 New Employees

        • Mentorship programs

        • Training support

        • Strong onboarding
        """)