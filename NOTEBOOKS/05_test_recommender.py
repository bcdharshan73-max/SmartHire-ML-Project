import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity

jobs = pd.read_csv("DATA/PROCESSED/jobs_cleaned.csv")

vectorizer = joblib.load("MODELS/job_vectorizer.pkl")
job_vectors = joblib.load("MODELS/job_vectors.pkl")

resume_text = """
Python SQL machine learning data analysis pandas numpy
scikit-learn statistics data visualization Power BI
"""

resume_vector = vectorizer.transform([resume_text])

similarity_scores = cosine_similarity(
    resume_vector,
    job_vectors
).flatten()

top_indices = similarity_scores.argsort()[-10:][::-1]

results = jobs.iloc[top_indices].copy()

results["Match_Score"] = similarity_scores[top_indices] * 100

print("\nTOP 10 MATCHING JOBS\n")

for rank, (_, row) in enumerate(results.iterrows(), start=1):
    print("Rank:", rank)
    print("Job:", row["Job_Title"])
    print("Company:", row["Company"])
    print("Location:", row["Location"])
    print("Match Score:", round(row["Match_Score"], 2), "%")
    print("-" * 60)