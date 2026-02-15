import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "Data/spam.csv"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(DATA_PATH, encoding="latin-1")

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Text Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=3000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# Model Training
# -----------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -----------------------------
# Model Evaluation
# -----------------------------
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Save Model & Vectorizer
# -----------------------------
with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model and vectorizer saved successfully!")
