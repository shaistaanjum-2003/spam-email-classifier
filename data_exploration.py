
import pandas as pd

df = pd.read_csv("data/spam_ham_dataset.csv")

print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns)

print("\nFirst 5 Emails:")
print(df.head())

print("\nSpam and Ham Counts:")
print(df["label"].value_counts())


print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nLabel Mapping:")
print(df.groupby("label")["label_num"].unique())


# Keep only the columns needed for our project
df = df[["text", "label"]]

print("\nUpdated Dataset Shape:", df.shape)
print("\nSample Email:")
print(df["text"].iloc[0])

print("\nTarget Distribution:")
print(df["label"].value_counts())
