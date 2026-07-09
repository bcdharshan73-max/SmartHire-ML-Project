import pandas as pd
import re
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("DATA/RAW/UpdatedResumeDataSet.csv")

def clean_text(text):
    text = re.sub(r"http\S+", " ", str(text))
    text = re.sub(r"RT|cc", " ", text)
    text = re.sub(r"#\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

df["Clean_Resume"] = df["Resume"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["Clean_Resume"],
    df["Category"],
    test_size=0.2,
    random_state=42,
    stratify=df["Category"]
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)

print("Total resumes:", len(df))
print("Number of categories:", df["Category"].nunique())
print("Training resumes:", len(X_train))
print("Testing resumes:", len(X_test))
print("\nAccuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

os.makedirs("MODELS", exist_ok=True)

joblib.dump(model, "MODELS/resume_classifier.pkl")
joblib.dump(vectorizer, "MODELS/resume_vectorizer.pkl")

print("\nSUCCESS!")
print("Classifier and vectorizer saved in MODELS folder.")