import streamlit as st
import pandas as pd
import joblib
import re
import sys
from pathlib import Path
from pypdf import PdfReader
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

jobs = pd.read_csv(ROOT / "DATA" / "PROCESSED" / "jobs_cleaned.csv")

classifier = joblib.load(ROOT / "MODELS" / "resume_classifier.pkl")
resume_vectorizer = joblib.load(ROOT / "MODELS" / "resume_vectorizer.pkl")
job_vectorizer = joblib.load(ROOT / "MODELS" / "job_vectorizer.pkl")
job_vectors = joblib.load(ROOT / "MODELS" / "job_vectors.pkl")

skill_list = [
    "python", "sql", "machine learning", "deep learning",
    "data analysis", "data analytics", "pandas", "numpy",
    "scikit-learn", "power bi", "tableau", "excel",
    "statistics", "data visualization", "tensorflow",
    "pytorch", "aws", "azure", "spark", "hadoop",
    "java", "javascript", "html", "css", "react",
    "docker", "kubernetes", "git", "mongodb", "mysql"
]

def clean_text(text):
    text = re.sub(r"http\S+", " ", str(text))
    text = re.sub(r"RT|cc", " ", text)
    text = re.sub(r"#\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

def extract_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "

    return text

def extract_docx(file):
    document = Document(file)
    return " ".join(paragraph.text for paragraph in document.paragraphs)

def extract_resume_text(file):
    if file.name.lower().endswith(".pdf"):
        return extract_pdf(file)

    if file.name.lower().endswith(".docx"):
        return extract_docx(file)

    if file.name.lower().endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    return ""

def extract_skills(text):
    text = str(text).lower()
    found_skills = []

    for skill in skill_list:
        if re.search(r"\b" + re.escape(skill) + r"\b", text):
            found_skills.append(skill)

    return set(found_skills)

def recommend_jobs(resume_text, top_n=10):
    resume_vector = job_vectorizer.transform([resume_text])

    scores = cosine_similarity(
        resume_vector,
        job_vectors
    ).flatten()

    top_indices = scores.argsort()[-top_n:][::-1]

    results = jobs.iloc[top_indices].copy()
    results["Match_Score"] = scores[top_indices] * 100

    return results

st.set_page_config(
    page_title="SmartHire",
    page_icon="💼",
    layout="wide"
)

st.title("💼 SmartHire")
st.subheader("Resume-to-Job Matching & Career Guidance Engine")

st.write(
    "Upload your resume to discover your predicted career category, "
    "matching jobs and missing skills."
)

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx", "txt"]
)

if uploaded_file is not None:

    resume_text = extract_resume_text(uploaded_file)

    if not resume_text.strip():
        st.error("Could not extract text from the uploaded resume.")

    else:
        st.success("Resume uploaded and analyzed successfully!")

        cleaned_resume = clean_text(resume_text)

        resume_features = resume_vectorizer.transform(
            [cleaned_resume]
        )

        predicted_category = classifier.predict(
            resume_features
        )[0]

        st.header("🎯 Predicted Career Category")
        st.success(predicted_category)

        resume_skills = extract_skills(resume_text)

        st.header("🛠 Skills Found in Resume")

        if resume_skills:
            st.write(", ".join(sorted(resume_skills)))
        else:
            st.warning("No skills from the current skill database were detected.")

        st.header("🔍 Top Matching Jobs")

        results = recommend_jobs(resume_text)

        for rank, (_, row) in enumerate(
            results.iterrows(),
            start=1
        ):
            with st.expander(
                f"{rank}. {row['Job_Title']} — {row['Match_Score']:.2f}% Match"
            ):
                st.write("**Company:**", row["Company"])
                st.write("**Location:**", row["Location"])
                st.write("**Experience:**", row["Experience"])

                job_skills = extract_skills(row["Skills"])

                matched_skills = resume_skills.intersection(
                    job_skills
                )

                missing_skills = job_skills.difference(
                    resume_skills
                )

                st.write("**Matched Skills:**")

                if matched_skills:
                    st.write(", ".join(sorted(matched_skills)))
                else:
                    st.write("No direct skill matches detected.")

                st.write("**Missing Skills:**")

                if missing_skills:
                    st.write(", ".join(sorted(missing_skills)))
                else:
                    st.write("No major missing skills detected.")

        st.header("📈 Career Guidance")

        top_job_skills = set()

        for _, row in results.head(5).iterrows():
            top_job_skills.update(
                extract_skills(row["Skills"])
            )

        overall_missing = top_job_skills.difference(
            resume_skills
        )

        if overall_missing:
            st.write(
                "To improve your chances for the top matching jobs, consider learning:"
            )

            for skill in sorted(overall_missing):
                st.write("•", skill.title())
        else:
            st.success(
                "Your detected skills match the major requirements of the top jobs."
            )