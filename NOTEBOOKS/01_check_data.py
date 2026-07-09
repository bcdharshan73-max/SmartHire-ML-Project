import pandas as pd
import os

raw_folder = "DATA/RAW"

print("FILES FOUND:\n")

for file in os.listdir(raw_folder):
    print(file)

print("\nCHECKING CSV FILES:\n")

for file in os.listdir(raw_folder):
    if file.endswith(".csv"):
        path = os.path.join(raw_folder, file)

        print("=" * 60)
        print("FILE:", file)

        df = pd.read_csv(path)

        print("SHAPE:", df.shape)
        print("COLUMNS:", df.columns.tolist())
        print("\nFIRST 3 ROWS:")
        print(df.head(3))
        print()