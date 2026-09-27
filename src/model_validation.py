import pandas as pd
from pathlib import Path

# Βρίσκει τον φάκελο fraud-risk-analytics
BASE_DIR = Path(__file__).resolve().parent

# Dataset
df = pd.read_csv(BASE_DIR / "creditcard.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print(df["Class"].value_counts())