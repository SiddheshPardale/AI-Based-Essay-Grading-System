import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import clean_text
from tqdm import tqdm

# Load dataset
df = pd.read_csv("dataset/ASAP2_train_sourcetexts.csv")

print("Dataset Loaded Successfully!")
print("Total Essays:", len(df))

tqdm.pandas()

# Clean essays
df["clean_text"] = df["full_text"].progress_apply(clean_text)

# Word count
df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))

# Character count
df["char_count"] = df["clean_text"].apply(len)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["clean_text"])

print("\nFeature Extraction Completed!")

print("TF-IDF Shape:", X.shape)

print(df[["score", "word_count", "char_count"]].head())


# Save processed dataset
df.to_csv("output/processed_dataset.csv", index=False)

print("\nProcessed dataset saved successfully!")