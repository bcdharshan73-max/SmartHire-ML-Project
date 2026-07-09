import pandas as pd
import os

raw_folder = "DATA/RAW"

df1 = pd.read_csv(os.path.join(raw_folder, "NaukriData_data analytics.csv"))
df2 = pd.read_csv(os.path.join(raw_folder, "NaukriData_Data Science.csv"))
df3 = pd.read_csv(os.path.join(raw_folder, "Naukri_Data_Scientist_and_Data_Analytics_Jobs_Data.csv"))

print("Dataset 1:", df1.shape)
print("Dataset 2:", df2.shape)
print("Dataset 3:", df3.shape)

df1 = df1[["Job_Titles", "Company_Names", "Experience_Required", "Locations", "Skills"]]
df2 = df2[["Job_Titles", "Company_Names", "Experience_Required", "Locations", "Skills"]]
df3 = df3[["Job Titles", "Company Names", "Experience Required", "Locations", "Skills"]]

common_columns = ["Job_Title", "Company", "Experience", "Location", "Skills"]

df1.columns = common_columns
df2.columns = common_columns
df3.columns = common_columns

jobs = pd.concat([df1, df2, df3], ignore_index=True)

print("Total jobs after merging:", jobs.shape)

jobs = jobs.drop_duplicates()
print("Jobs after removing duplicates:", jobs.shape)

jobs = jobs.dropna(subset=["Job_Title", "Skills"])
print("Jobs after removing missing values:", jobs.shape)

for column in common_columns:
    jobs[column] = jobs[column].fillna("").astype(str).str.strip()

jobs["Job_Text"] = jobs["Job_Title"] + " " + jobs["Skills"]

os.makedirs("DATA/PROCESSED", exist_ok=True)

jobs.to_csv("DATA/PROCESSED/jobs_cleaned.csv", index=False)

print("\nFINAL DATASET")
print("Shape:", jobs.shape)
print("Columns:", jobs.columns.tolist())
print(jobs.head())
print("\nSUCCESS!")