# SmartHire

SmartHire is a Resume-to-Job Matching and Career Guidance System built using Machine Learning.

## Features

- Upload resume in PDF, DOCX, or TXT format
- Predict resume career category
- Extract skills from resume
- Recommend top matching jobs
- Calculate job similarity scores
- Identify matched and missing skills
- Provide career guidance

## Machine Learning Techniques

### Supervised Learning

Resume classification using:

- TF-IDF Vectorization
- Logistic Regression

Model Accuracy: 99.48%

### Unsupervised Learning

Job recommendation using:

- TF-IDF Vectorization
- Cosine Similarity

## Dataset

Resume Dataset:

- 962 resumes
- 25 categories

Job Dataset:

- 86,904 raw job records
- 55,525 cleaned job records

## Project Structure

SMARTHIRE/
- APPS/
- DATA/
- MODELS/
- NOTEBOOKS/
- REPORTS/
- README.md
- requirements.txt

## How to Run

Install required libraries:

pip install -r requirements.txt

Run the application:

python -m streamlit run APPS/streamlit_app.py

## Output

The system displays:

- Predicted Career Category
- Skills Found in Resume
- Top 10 Matching Jobs
- Match Scores
- Matched Skills
- Missing Skills
- Career Guidance

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Cosine Similarity
- Streamlit

## Author

Dharshan Gowda B.C