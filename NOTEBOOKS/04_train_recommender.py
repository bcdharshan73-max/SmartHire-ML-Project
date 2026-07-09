import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer

jobs = pd.read_csv("DATA/PROCESSED/jobs_cleaned.csv")

jobs["Job_Text"] = jobs["Job_Text"].fillna("").astype(str)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2)
)

job_vectors = vectorizer.fit_transform(jobs["Job_Text"])

print("Total jobs:", len(jobs))
print("Job vector shape:", job_vectors.shape)

os.makedirs("MODELS", exist_ok=True)

joblib.dump(vectorizer, "MODELS/job_vectorizer.pkl")
joblib.dump(job_vectors, "MODELS/job_vectors.pkl")

print("\nSUCCESS!")
print("Job recommender files saved in MODELS folder.")