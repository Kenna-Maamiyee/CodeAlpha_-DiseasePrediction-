 # 🏥 AI Hospital System (Heart Disease Prediction)

An AI-powered hospital management and diagnosis system built with Streamlit and Machine Learning.  
It provides heart disease prediction, user authentication, admin dashboard, patient history tracking, PDF medical reports, and AI explainability.

---

## 🚀 Features

### 🔐 Authentication System
- Sign Up / Login system
- Role-based access (Doctor, Nurse, Admin)
- Forgot password recovery

### 🧠 AI Disease Prediction
- Heart disease prediction using Machine Learning
- Real-time patient input form
- Risk probability score

### 📊 Explainable AI (SHAP)
- Feature importance visualization
- Model explanation for medical decisions

### 🏥 Patient Management
- Store patient history in SQLite database
- View past predictions
- Admin can view all users

### 📄 PDF Medical Reports
- Generate hospital-style PDF reports
- Download patient diagnosis reports
- Professional medical formatting

### 🤖 AI Chatbot Assistant
- Basic AI medical assistant (GPT-ready integration)

---

## 🛠️ Tech Stack

- Python 🐍
- Streamlit 🎈
- Scikit-learn 🤖
- XGBoost ⚡
- SHAP 🧠
- SQLite 🗄️
- ReportLab 📄
- Pandas / NumPy 📊

---

## 📂 Project Structure

```

Disease_Prediction/
│
├── app.py                 # Main Streamlit application
├── heart_model.pkl       # Trained ML model
├── patients.db           # SQLite database (auto-created)
├── requirements.txt      # Dependencies
└── README.md             # Project documentation

````

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone :https://github.com/Kenna-Maamiyee/CodeAlpha_-DiseasePrediction-.git 
cd ai-hospital-system
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app

```bash
streamlit run app.py
```

 

## 🌐 Deployment

You can deploy this app using:

* Streamlit Cloud (Recommended)
* Render
* AWS / Azure (Advanced)

---

## 🧪 Example Inputs

* Age: 45
* BP: 120
* Cholesterol: 210
* Chest Pain Type: 1

---

## ⚠️ Disclaimer

This system is for **educational purposes only**.
It is NOT a real medical diagnostic tool.

 

## 👨‍⚕️ Author

Developed by: Yohannes Alemayehu

   
  