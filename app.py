# =========================
# 🏥 AI HOSPITAL SYSTEM
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import sqlite3
import hashlib
import uuid
import os
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Hospital System", layout="wide")

# =========================
# LOAD MODEL
# =========================
model = joblib.load("heart_model.pkl")
features = model.feature_names_in_

explainer = shap.TreeExplainer(model)

# =========================
# DATABASES
# =========================

# --- Patients DB ---
conn = sqlite3.connect("patients.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age REAL,
    sex INTEGER,
    chest_pain REAL,
    bp REAL,
    cholesterol REAL,
    fbs INTEGER,
    ekg REAL,
    max_hr REAL,
    angina INTEGER,
    st_dep REAL,
    slope REAL,
    vessels REAL,
    thallium REAL,
    prediction INTEGER,
    probability REAL,
    time TEXT
)
""")

# --- Users DB ---
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT,
    role TEXT DEFAULT 'Doctor',
    reset_token TEXT DEFAULT NULL
)
""")

conn.commit()

# =========================
# AUTH FUNCTIONS
# =========================
def make_hash(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def add_user(username, password, role):
    cursor.execute("INSERT INTO users (username, password, role) VALUES (?,?,?)",
                   (username, make_hash(password), role))
    conn.commit()

def login_user(username, password):
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?",
                   (username, make_hash(password)))
    return cursor.fetchone()

def get_all_users():
    cursor.execute("SELECT id, username, role FROM users")
    return cursor.fetchall()

def create_reset_token(username):
    token = str(uuid.uuid4())
    cursor.execute("UPDATE users SET reset_token=? WHERE username=?",
                   (token, username))
    conn.commit()
    return token

def reset_password(token, new_password):
    cursor.execute("UPDATE users SET password=?, reset_token=NULL WHERE reset_token=?",
                   (make_hash(new_password), token))
    conn.commit()

# =========================
# SESSION STATE
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = ""
if "role" not in st.session_state:
    st.session_state.role = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================
# GPT CHATBOT (OPTIONAL SAFE)
# =========================
def medical_bot(user_input, prediction=None, probability=None):
    return f"""
    🤖 AI Assistant Response:

    You asked: {user_input}

    Current patient risk: {prediction}
    Probability: {probability}

    ⚠ This is AI support only, not medical diagnosis.
    """

# =========================
# LOGIN / SIGNUP UI
# =========================
if not st.session_state.logged_in:

    st.title("🏥 AI Hospital Login System")

    menu = ["Login", "Sign Up", "Forgot Password"]
    choice = st.selectbox("Select Option", menu)

    # ---------------- LOGIN ----------------
    if choice == "Login":
        st.subheader("🔐 Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            result = login_user(username, password)

            if result:
                st.session_state.logged_in = True
                st.session_state.user = result[1]
                st.session_state.role = result[3]
                st.success(f"Welcome {result[1]} ({result[3]})")
                st.rerun()
            else:
                st.error("Invalid credentials")

    # ---------------- SIGN UP ----------------
    elif choice == "Sign Up":
        st.subheader("📝 Create Account")

        username = st.text_input("New Username")
        password = st.text_input("New Password", type="password")
        role = st.selectbox("Role", ["Doctor", "Nurse", "Admin"])

        if st.button("Create Account"):
            try:
                add_user(username, password, role)
                st.success("Account created successfully")
            except:
                st.error("Username already exists")

    # ---------------- FORGOT PASSWORD ----------------
    elif choice == "Forgot Password":
        st.subheader("🔑 Reset Password")

        username = st.text_input("Username")

        if st.button("Generate Reset Token"):
            token = create_reset_token(username)
            st.info(f"Your reset token: {token}")

        token_input = st.text_input("Enter Token")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Reset Password"):
            reset_password(token_input, new_pass)
            st.success("Password updated")

    st.stop()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🏥 Hospital Dashboard")
st.sidebar.write(f"User: {st.session_state.user}")
st.sidebar.write(f"Role: {st.session_state.role}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# =========================
# ADMIN DASHBOARD
# =========================
if st.session_state.role == "Admin":
    st.subheader("🟣 Admin Panel - Users")

    users = get_all_users()
    df_users = pd.DataFrame(users, columns=["ID", "Username", "Role"])

    st.dataframe(df_users, width="stretch")

st.write("---")

# =========================
# PATIENT INPUT
# =========================
st.subheader("🧠 Heart Disease Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    Age = st.slider("Age", 20, 80, 40)
    Sex = st.selectbox("Sex", ["Male", "Female"])
    Sex = 1 if Sex == "Male" else 0
    ChestPain = st.slider("Chest Pain Type", 0, 3, 1)

with col2:
    BP = st.slider("BP", 80, 200, 120)
    Chol = st.slider("Cholesterol", 100, 400, 200)
    MaxHR = st.slider("Max HR", 60, 200, 150)

with col3:
    FBS = st.selectbox("FBS > 120", [0, 1])
    EKG = st.slider("EKG", 0, 2, 1)
    Angina = st.selectbox("Exercise Angina", [0, 1])
    STDep = st.slider("ST Depression", 0.0, 6.0, 1.0)
    Slope = st.slider("Slope", 0, 2, 1)
    Vessels = st.slider("Vessels", 0, 3, 0)
    Thallium = st.slider("Thallium", 0, 3, 2)

input_df = pd.DataFrame([[
    Age, Sex, ChestPain, BP, Chol, FBS,
    EKG, MaxHR, Angina, STDep,
    Slope, Vessels, Thallium
]], columns=features)


# 🔵 STEP 3 — PDF GENERATOR FUNCTION

def generate_pdf(data, prediction, probability, filename="medical_report.pdf"):

    doc = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    # =========================
    # CUSTOM STYLES
    # =========================
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.darkblue,
        alignment=1
    )

    normal_style = styles["Normal"]

    risk_style_high = ParagraphStyle(
        "RiskHigh",
        fontSize=14,
        textColor=colors.red
    )

    risk_style_low = ParagraphStyle(
        "RiskLow",
        fontSize=14,
        textColor=colors.green
    )

    content = []

    # =========================
    # HEADER
    # =========================
    content.append(Paragraph("🏥 CITY HOSPITAL AI REPORT", title_style))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Patient Medical Summary", styles["Heading2"]))
    content.append(Spacer(1, 12))

    # =========================
    # PATIENT DATA TABLE
    # =========================
    table_data = [
        ["Age", data["Age"]],
        ["Sex", data["Sex"]],
        ["Blood Pressure", data["BP"]],
        ["Cholesterol", data["Chol"]],
    ]

    table = Table(table_data)

    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightblue),
        ("TEXTCOLOR", (0,0), (-1,0), colors.black),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 11),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))

    content.append(table)
    content.append(Spacer(1, 15))

    # =========================
    # PREDICTION RESULT
    # =========================
    if prediction == 1:
        result_text = "⚠ HIGH RISK OF HEART DISEASE"
        style = risk_style_high
    else:
        result_text = "✅ LOW RISK OF HEART DISEASE"
        style = risk_style_low

    content.append(Paragraph("Diagnosis Result:", styles["Heading2"]))
    content.append(Paragraph(result_text, style))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Risk Probability: {round(probability*100, 2)}%", normal_style))
    content.append(Spacer(1, 20))

    # =========================
    # FOOTER DISCLAIMER
    # =========================
    content.append(Paragraph(
        "⚠ This AI-generated report is for educational purposes only and not a medical diagnosis.",
        styles["Italic"]
    ))

    doc.build(content)

    return filename



# =========================
# PREDICTION
# =========================
# =========================
# PREDICTION
# =========================
if st.button("🧠 Predict"):

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    prediction_clean = 1 if str(prediction) in ["1", "Presence", "Yes"] else 0

    if prediction_clean == 1:
        st.error("⚠ High Risk Detected")
    else:
        st.success("✅ Low Risk Detected")

    st.progress(float(proba[1]))

    report_data = {
        "Age": Age,
        "Sex": Sex,
        "BP": BP,
        "Chol": Chol
    }

    pdf_file = generate_pdf(report_data, prediction_clean, float(proba[1]))

    with open(pdf_file, "rb") as f:
        st.download_button(
            "📄 Download Medical Report PDF",
            f,
            file_name="medical_report.pdf"
        )

    # =========================
    # SAVE PATIENT
    # =========================
    cursor.execute("""
    INSERT INTO history VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        Age, Sex, ChestPain, BP, Chol, FBS,
        EKG, MaxHR, Angina, STDep,
        Slope, Vessels, Thallium,
        prediction_clean,
        float(proba[1]),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()

    # =========================
    # SHAP
    # =========================
    st.subheader("🧠 Explainability (SHAP)")

    shap_values = explainer.shap_values(input_df)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.array(shap_values)[0].flatten()

    min_len = min(len(features), len(shap_values))

    fig, ax = plt.subplots()
    ax.barh(features[:min_len], shap_values[:min_len])
    st.pyplot(fig)

# =========================
# HISTORY
# =========================
st.write("---")
st.subheader("🔵 Patient History")

df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)
st.dataframe(df, width="stretch")

if st.button("🗑 Clear History"):
    cursor.execute("DELETE FROM history")
    conn.commit()
    st.success("Cleared!")


import os
import joblib

model_path = os.path.join(os.path.dirname(__file__), "heart_model.pkl")
model = joblib.load(model_path)