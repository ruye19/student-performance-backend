from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="Student Performance Prediction API")

# Load models
logistic_model = joblib.load("logistic_model.pkl")
decision_tree_model = joblib.load("decision_tree_model.pkl")
encoder = joblib.load("encoder.pkl")


@app.get("/")
def home():
    return {"message": "Student Performance Prediction API is running"}


@app.post("/predict/logistic")
def predict_logistic(data: dict):
    """
    Predict student performance using Logistic Regression
    """
    values = np.array(list(data.values())).reshape(1, -1)
    prediction = logistic_model.predict(values)[0]

    return {
        "model": "Logistic Regression",
        "prediction": "PASS" if prediction == 1 else "FAIL"
    }


@app.post("/predict/decision-tree")
def predict_decision_tree(data: dict):
    """
    Predict student performance using Decision Tree
    """
    values = np.array(list(data.values())).reshape(1, -1)
    prediction = decision_tree_model.predict(values)[0]

    return {
        "model": "Decision Tree",
        "prediction": "PASS" if prediction == 1 else "FAIL"
    }
