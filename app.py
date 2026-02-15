from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# -----------------------------
# Load Model & Vectorizer
# -----------------------------
MODEL_PATH = "model/model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = ""

    if request.method == "POST":
        message = request.form.get("message")

        if message:
            transformed_msg = vectorizer.transform([message])
            prediction = model.predict(transformed_msg)[0]

            if prediction == 1:
                prediction_text = "🚨 SPAM MESSAGE"
            else:
                prediction_text = "✅ NOT SPAM MESSAGE"

    return render_template("index.html", result=prediction_text)

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
