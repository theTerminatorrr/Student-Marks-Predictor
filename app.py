"""Backend for the Student Mark Predictor frontend.

Run:
    pip install flask pandas scikit-learn
    python app.py
Then open http://127.0.0.1:5000
"""
import os
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "student_performance_messy_expanded.csv")
FEATURES = ["study_hours", "attendance_percent", "previous_score", "sleep_hours"]

# Allowed input ranges (must match the sliders in index.html)
LIMITS = {
    "study_hours": (0, 24),
    "attendance_percent": (0, 100),
    "previous_score": (0, 100),
    "sleep_hours": (0, 24),
}


def train_model():
    """Same cleaning + training steps as student_mark_predict.py (no plots)."""
    df = pd.read_csv(CSV_PATH)
    df = df.drop_duplicates()
    df["study_hours"] = pd.to_numeric(
        df["study_hours"].astype(str).str.replace(" hrs", ""), errors="coerce")
    df["attendance_percent"] = pd.to_numeric(
        df["attendance_percent"].astype(str).str.replace("eighty", "80").str.replace("%", ""),
        errors="coerce")
    df["previous_score"] = pd.to_numeric(
        df["previous_score"].astype(str).str.replace(" / 100", ""), errors="coerce")
    df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors="coerce")

    df.loc[df["study_hours"] > 24, "study_hours"] = None
    df.loc[df["attendance_percent"] > 100, "attendance_percent"] = None
    df.loc[df["previous_score"] > 100, "previous_score"] = None
    df.loc[df["sleep_hours"] < 0, "sleep_hours"] = None
    df = df[df["exam_score"] <= 100]

    X, y = df[FEATURES], df["exam_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    medians = X_train.median()
    X_train, X_test = X_train.fillna(medians), X_test.fillna(medians)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    return model, mae


model, MAE = train_model()
print(f"Model trained. Test MAE: {MAE:.2f}")

app = Flask(__name__)


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/api/info")
def info():
    return jsonify(mae=round(MAE, 2))


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    values = {}
    for name, (lo, hi) in LIMITS.items():
        try:
            v = float(data[name])
        except (KeyError, TypeError, ValueError):
            return jsonify(error=f"'{name}' is missing or not a number."), 400
        if not lo <= v <= hi:
            return jsonify(error=f"'{name}' must be between {lo} and {hi}."), 400
        values[name] = v

    row = pd.DataFrame([values], columns=FEATURES)
    score = float(model.predict(row)[0])
    return jsonify(score=round(max(0.0, min(100.0, score)), 1), mae=round(MAE, 2))


if __name__ == "__main__":
    app.run(debug=False)
