import pandas as pd
import re
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

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

_, X_test, _, y_test = train_test_split(
    df["Clean_Resume"],
    df["Category"],
    test_size=0.2,
    random_state=42,
    stratify=df["Category"]
)

model = joblib.load("MODELS/resume_classifier.pkl")
vectorizer = joblib.load("MODELS/resume_vectorizer.pkl")

X_test_tfidf = vectorizer.transform(X_test)
predictions = model.predict(X_test_tfidf)

os.makedirs("REPORTS", exist_ok=True)

report = classification_report(
    y_test,
    predictions,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv("REPORTS/classification_report.csv")

plt.figure(figsize=(12, 8))
df["Category"].value_counts().plot(kind="bar")
plt.title("Resume Category Distribution")
plt.xlabel("Category")
plt.ylabel("Number of Resumes")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("REPORTS/category_distribution.png", dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(18, 18))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    xticks_rotation=90,
    ax=ax,
    cmap="Blues"
)

plt.title("Resume Classifier Confusion Matrix")
plt.tight_layout()
plt.savefig("REPORTS/confusion_matrix.png", dpi=300)
plt.close()

print("SUCCESS!")
print("Created:")
print("REPORTS/classification_report.csv")
print("REPORTS/category_distribution.png")
print("REPORTS/confusion_matrix.png")