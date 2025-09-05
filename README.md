# 🎓 AI-Based Student Dropout Prediction & Counseling System

## 📌 Overview
This project is a prototype of an **AI-powered system** that predicts student dropout risk and provides a dashboard for counselors to monitor students.

It combines:
- 🤖 **Machine Learning (XGBoost)** for dropout prediction  
- 🌐 **Flask API** for model serving  
- 📊 **Interactive Dashboard** (table, chart, filters, search) for counselors  

---

## 📂 Project Structure
ai_dropout_prototype/
│── app/
│ ├── app.py # Flask app (API + Dashboard)
│ └── templates/dashboard.html
│── data/sample_students.csv # Sample dataset
│── models/model_pipeline.joblib # Trained ML model
│── src/train_pipeline.py # Model training script
│── requirements.txt # Dependencies
│── README.md # Project guide

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt

python src/train_pipeline.py

python app/app.py

then open your https link
