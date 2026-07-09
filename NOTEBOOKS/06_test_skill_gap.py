import pandas as pd
import re

jobs = pd.read_csv("DATA/PROCESSED/jobs_cleaned.csv")

skill_list = [
    "python", "sql", "machine learning", "deep learning",
    "data analysis", "data analytics", "pandas", "numpy",
    "scikit-learn", "power bi", "tableau", "excel",
    "statistics", "data visualization", "tensorflow",
    "pytorch", "aws", "azure", "spark", "hadoop",
    "java", "javascript", "html", "css", "react",
    "docker", "kubernetes", "git", "mongodb", "mysql"
]

def extract_skills(text):
    text = str(text).lower()
    found_skills = []

    for skill in skill_list:
        if re.search(r"\b" + re.escape(skill) + r"\b", text):
            found_skills.append(skill)

    return set(found_skills)

resume_text = """
Python SQL machine learning data analysis pandas numpy
scikit-learn statistics
"""

job_skills_text = """
Python SQL machine learning data analysis pandas numpy
Power BI Tableau AWS
"""

resume_skills = extract_skills(resume_text)
job_skills = extract_skills(job_skills_text)

matched_skills = resume_skills.intersection(job_skills)
missing_skills = job_skills.difference(resume_skills)

print("\nRESUME SKILLS:")
print(", ".join(sorted(resume_skills)))

print("\nJOB REQUIRED SKILLS:")
print(", ".join(sorted(job_skills)))

print("\nMATCHED SKILLS:")
print(", ".join(sorted(matched_skills)))

print("\nMISSING SKILLS:")
print(", ".join(sorted(missing_skills)))