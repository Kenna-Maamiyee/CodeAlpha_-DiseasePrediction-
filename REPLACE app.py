import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Hospital AI System",
    page_icon="🏥",
    layout="wide"
)

# =========================================
# LOAD MODEL
# =========================================
model = joblib.load("heart_model.pkl")
features = joblib.load("features.pkl")

# =========================================
# CSS DESIGN (HOSPITAL STYLE)
# =========================================
st.markdown("""
<style>

body {
    background-color: #f5f7fb;
}

.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #0b3d91;
    text-align: center;
}

.sub-title {
    text-align: center;
    color: gray;
    font-size: 16px;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
}

.risk-high {
    padding: 15px;
    background-color: #ffebee;
    color: #c62828;
    border-radius: 10px;
    font-weight: bold;
    text-align: center;
}

.risk-low {
    padding: 15px;
    background-color: #e8f5e9;
    color: #2e7d32;
    border-radius: 10px;
    font-weight: bold;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# HEADER
# =========================================
st.markdown("<div class='main-title'>🏥 Hospital AI Diagnostic System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Clinical Decision Support for Heart Disease Prediction</div>", unsafe_allow_html=True)

st.write("---")

# =========================================
# LAYOUT
# =========================================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧍 Patient Clinical Data")

    Age = st.slider("Age", 20, 80, 40)
    Sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    ChestPain = st.slider("Chest Pain Type", 0, 3, 1)
    BP = st.slider("Blood Pressure (mmHg)", 80, 200, 120)
    Chol = st.slider("Cholesterol (mg/dL)", 100, 400, 200)
    FBS = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

with col2:
    st.markdown("### ❤️ Cardiac Test Results")

    EKG = st.slider("ECG Results", 0, 2, 1)
    MaxHR = st.slider("Max Heart Rate", 60, 200, 150)
    ExerciseAngina = st.selectbox("Exercise Angina", [0, 1])
    STDep = st.slider("ST Depression", 0.0, 6.0, 1.0)
    Slope = st.slider("ST Slope", 0, 2, 1)
    Vessels = st.slider("Fluoroscopy Vessels", 0, 3, 0)
    Thallium = st.slider("Thallium Stress Test", 0, 3, 2)

st.write("---")

# =========================================
# INPUT DATA
# =========================================
input_df = pd.DataFrame([[
    Age, Sex, ChestPain, BP, Chol, FBS,
    EKG, MaxHR, ExerciseAngina, STDep,
    Slope, Vessels, Thallium
]], columns=features)

# =========================================
# PREDICTION
# =========================================
if st.button("🩺 Run Clinical Diagnosis"):

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.write("---")

    # =========================================
    # RESULT DISPLAY (HOSPITAL STYLE)
    # =========================================
    if prediction == 1:
        st.markdown("<div class='risk-high'>⚠ HIGH RISK OF HEART DISEASE</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='risk-low'>✅ LOW RISK OF HEART DISEASE</div>", unsafe_allow_html=True)

    # =========================================
    # PROBABILITY
    # =========================================
    st.subheader("📊 Risk Probability")
    st.write({
        "No Disease": round(proba[0], 3),
        "Disease": round(proba[1], 3)
    })

    st.progress(float(proba[1]))

    # =========================================
    # AI EXPLANATION
    # =========================================
    st.write("---")
    st.subheader("🧠 Clinical AI Explanation")

    importance = model.feature_importances_

    exp_df = pd.DataFrame({
        "Feature": features,
        "Impact Score": importance
    }).sort_values("Impact Score")

    fig, ax = plt.subplots()
    ax.barh(exp_df["Feature"], exp_df["Impact Score"])
    ax.set_title("Key Medical Risk Factors")

    st.pyplot(fig)

    st.write("### 🔥 Top Risk Contributors")
    st.write(exp_df.sort_values("Impact Score", ascending=False).head(3))