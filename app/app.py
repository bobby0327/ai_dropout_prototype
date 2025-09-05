from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained pipeline
model_path = os.path.join("models", "model_pipeline.joblib")
model = joblib.load(model_path)

# Risk tier function
def risk_tier(score):
    if score > 0.7:
        return "High"
    elif score > 0.4:
        return "Medium"
    else:
        return "Low"

@app.route('/')
def dashboard():
    data_path = os.path.join("data", "sample_students.csv")
    df = pd.read_csv(data_path)

    # Run predictions
    probs = model.predict_proba(df.drop(columns=["dropout_within_1yr"]))[:, 1]
    df["risk_score"] = probs
    df["risk_tier"] = df["risk_score"].apply(risk_tier)

    # Count tiers for chart
    tier_counts = df["risk_tier"].value_counts().to_dict()

    return render_template(
        "dashboard.html",
        students=df.to_dict(orient="records"),
        tier_counts=tier_counts
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    df = pd.DataFrame([data])
    if "student_id" in df.columns:
        student_id = df["student_id"].iloc[0]
    else:
        student_id = "N/A"

    probs = model.predict_proba(df.drop(columns=["student_id"]))[:, 1]
    risk_score = float(probs[0])
    tier = risk_tier(risk_score)

    response = {
        "student_id": student_id,
        "risk_score": risk_score,
        "risk_tier": tier
    }

    return jsonify([response])

if __name__ == "__main__":
    print("🚀 Flask is starting on http://127.0.0.1:5000 ...")
    app.run(debug=True)
